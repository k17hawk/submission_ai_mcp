#!/usr/bin/env python3
"""
Streamlit Client for Insurance Claim MCP Server
Uses FastMCP Client to connect to the server and call tools.
Run with: streamlit run streamlit_client.py
"""

import asyncio
import streamlit as st
from fastmcp import Client

# ── Page setup ──────────────────
st.set_page_config(page_title="Claim Processor", layout="wide")
st.title("🏦 Insurance Claim Processing")

# ── Sidebar: server connection ──
st.sidebar.header("Server Connection")
server_url = st.sidebar.text_input(
    "MCP Server SSE URL",
    value="http://localhost:8000/sse",
    help="Default for `fastmcp dev`"
)

# ── Main form ───────────────────
with st.form("claim_form"):
    st.subheader("Claim Information")

    col1, col2 = st.columns(2)
    with col1:
        customer_name = st.text_input("Customer Name", value="Allison Hill")
        customer_id = st.text_input("Customer ID", value="CUST-93810")
    with col2:
        policy_number = st.text_input("Policy Number (optional, extracted from text)", disabled=True)

    claim_text = st.text_area(
        "Claim Description",
        height=200,
        value=(
            "Was involved in a 2-car pileup on I-95 at 2022-08-09 at FL Brownhaven "
            "and it has been 8 hours where i had Front Collision.\n"
            "My  2011 Chevrolet Equinox. Policy POL-651065.\n"
            "Name: Allison Hill with CUST-93810. 1 witnesses saw everything.\n"
            "Police report available. Damage around $8,000."
        ),
    )

    submitted = st.form_submit_button("🚀 Process Claim")

# ── Async client call ───────────
async def process_claim(text, name, cid):
    """Connect to the MCP server and run the complete pipeline."""
    try:
        async with Client(server_url) as client:
            result = await client.complete_pipeline(
                claim_text=text,
                customer_name=name,
                customer_id=cid
            )
            return result
    except Exception as e:
        return {"error": str(e)}

# ── Display results ─────────────
if submitted:
    with st.spinner("Processing claim... This may take a few seconds."):
        # Run async function inside Streamlit (sync wrapper)
        result = asyncio.run(process_claim(claim_text, customer_name, customer_id))

    if "error" in result and "Failed" not in result.get("error", ""):
        st.error(f"⚠️ Processing error: {result['error']}")
    else:
        st.success("✅ Claim processed successfully!")

        # Show decision and status
        decision = result.get("final_decision", "N/A")
        status = result.get("status", {}).get("value", "N/A") if isinstance(result.get("status"), dict) else result.get("status", "N/A")

        st.metric("Final Decision", decision)
        st.metric("Claim Status", status)
        st.metric("Processing Time (ms)", f"{result.get('processing_time_ms', 0):.0f}")

        # Full summary
        with st.expander("📄 Full Summary", expanded=True):
            st.text(result.get("summary", "No summary generated."))

        # Detailed JSON
        with st.expander("🔍 Raw JSON Response"):
            st.json(result)

# ── Individual tool testing (optional) ──
with st.expander("⚙️ Test Individual Tools"):
    tool = st.selectbox("Choose tool", ["parse_claim", "lookup_policy", "check_risk", "build_features", "detect_fraud", "health_check"])
    if st.button("Run Tool"):
        async def call_tool():
            try:
                async with Client(server_url) as client:
                    if tool == "parse_claim":
                        return await client.parse_claim(text=claim_text)
                    elif tool == "lookup_policy":
                        # Need to extract policy number first (simplified)
                        return await client.lookup_policy(
                            policy_number="POL-651065",
                            incident_date="2022-08-09",
                            incident_type="Front Collision",
                            claim_amount=8000.0
                        )
                    elif tool == "health_check":
                        return await client.health_check()
                    # ... add other tools as needed
                    else:
                        return {"error": f"Tool '{tool}' not implemented in test UI"}
            except Exception as e:
                return {"error": str(e)}

        tool_result = asyncio.run(call_tool())
        st.json(tool_result)