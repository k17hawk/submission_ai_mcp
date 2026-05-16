import asyncio
import streamlit as st
from fastmcp import Client

st.set_page_config(page_title="Claim Processor", layout="wide")
st.title("🏦 Insurance Claim Processing")

server_url = st.sidebar.text_input("MCP Server SSE URL", value="http://localhost:8000/sse")

with st.form("claim_form"):
    customer_name = st.text_input("Customer Name", value="Allison Hill")
    customer_id = st.text_input("Customer ID", value="CUST-93810")
    claim_text = st.text_area("Claim Description", value=(
        "Was involved in a 2-car pileup on I-95 at 2022-08-09 at FL Brownhaven "
        "and it has been 8 hours where i had Front Collision.\n"
        "My  2011 Chevrolet Equinox. Policy POL-651065.\n"
        "Name: Allison Hill with CUST-93810. 1 witnesses saw everything.\n"
        "Police report available. Damage around $8,000."
    ))
    submitted = st.form_submit_button("🚀 Process Claim")

async def process_claim(text, name, cid):
    try:
        async with Client(server_url) as client:
            result = await client.call_tool(
                "complete_pipeline",
                {"claim_text": text, "customer_name": name, "customer_id": cid},
            )
            return result
    except Exception as e:
        return {"error": str(e)}

if submitted:
    with st.spinner("Processing claim..."):
        result = asyncio.run(process_claim(claim_text, customer_name, customer_id))

    if "error" in result:
        st.error(f"⚠️ {result['error']}")
    else:
        st.success("✅ Claim processed!")
        st.metric("Final Decision", result.get("final_decision", "N/A"))
        st.text("Summary:\n" + result.get("summary", ""))
        with st.expander("Raw JSON"):
            st.json(result)