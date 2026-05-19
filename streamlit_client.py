import asyncio
import streamlit as st
from fastmcp import Client

st.set_page_config(page_title="Claim Processor", layout="wide")
st.title("🏦 Insurance Claim Processing")

server_url = st.sidebar.text_input("MCP Server SSE URL", value="http://localhost:8000/sse")

with st.form("claim_form"):
    customer_name = st.text_input("Customer Name", value="Allison Hill")
    customer_id = st.text_input("Customer ID", value="CUST-93810")
    claim_text = st.text_area(
        "Claim Description",
        value=(
            "Was involved in a 2-car pileup on I-95 at 2022-08-09 at FL Brownhaven "
            "and it has been 8 hours where i had Front Collision.\n"
            "My  2011 Chevrolet Equinox. Policy POL-651065.\n"
            "Name: Allison Hill with CUST-93810. 1 witnesses saw everything.\n"
            "Police report available. Damage around $8,000."
        ),
        height=200,
    )
    submitted = st.form_submit_button("🚀 Process Claim")

async def process_claim(text, name, cid):
    try:
        async with Client(server_url) as client:
            result = await client.call_tool(
                "complete_pipeline",
                {"claim_text": text, "customer_name": name, "customer_id": cid},
            )
            import json
            raw = result.content[0].text
            # Debug: st.write("DEBUG raw:", raw[:500])
            clean = raw.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:-1])
            return json.loads(clean)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}

if submitted:
    with st.spinner("Processing claim..."):
        result = asyncio.run(process_claim(claim_text, customer_name, customer_id))

    if "error" in result:
        st.error(f"⚠️ {result['error']}")
    else:
        # Display main decision
        final_decision = result.get("final_decision", "N/A")
        if final_decision == "AUTO_APPROVE":
            st.success(f"✅ {final_decision}")
        elif final_decision in ["AUTO_DENY", "POLICY_NOT_FOUND"]:
            st.error(f"❌ {final_decision}")
        else:
            st.warning(f"⚠️ {final_decision}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Score", f"{result.get('risk_score', 0):.2f}")
        with col2:
            st.metric("Fraud Probability", f"{result.get('fraud_probability', 0):.2f}")
        with col3:
            st.metric("Status", result.get("status", "UNKNOWN"))

        # Show extracted data
        with st.expander("📄 Extracted Claim Data"):
            extracted = result.get("extracted_data", {})
            st.json(extracted)

        # Show policy data
        with st.expander("📋 Policy Data"):
            policy = result.get("policy_data", {})
            # Show only relevant fields
            policy_summary = {
                "Policy Number": policy.get("policy_number"),
                "Insured Name": policy.get("insured_name"),
                "Policy Status": policy.get("policy_status"),
                "Effective Date": policy.get("effective_date"),
                "Expiration Date": policy.get("expiration_date"),
                "Deductible": policy.get("deductible"),
                "Coverage Limit": policy.get("coverage_limit"),
                "Vehicle Make": policy.get("make"),
                "Vehicle Model": policy.get("model"),
                "Prior Claims Count": policy.get("prior_claims_count"),
            }
            st.json(policy_summary)

        # Show warnings and violations
        if result.get("risk_violations"):
            st.error("🚨 Risk Violations")
            for v in result["risk_violations"]:
                st.write(f"- **{v.get('rule_name')}**: {v.get('message')}")

        if result.get("risk_warnings"):
            st.warning("⚠️ Risk Warnings")
            for w in result["risk_warnings"]:
                st.write(f"- **{w.get('rule_name')}**: {w.get('message')}")

        if result.get("policy_warnings"):
            st.info("📌 Policy Warnings")
            for w in result["policy_warnings"]:
                st.write(f"- {w}")

        # Final recommendation
        st.subheader("💡 Recommended Action")
        st.write(result.get("recommended_action", "No recommendation"))

        with st.expander("🔍 Full JSON Response"):
            st.json(result)