import asyncio
import streamlit as st
from fastmcp import Client
import logging
from datetime import datetime
import json

st.set_page_config(page_title="Claim Processor", layout="wide")
st.title("🏦 Insurance Claim Processing")

# Initialize session state for logs
if 'logs' not in st.session_state:
    st.session_state.logs = []

# Custom log handler to capture logs for Streamlit
class StreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.setLevel(logging.DEBUG)
        
    def emit(self, record):
        try:
            # Format the log message
            log_entry = self.format(record)
            
            # Store logs in session state
            if 'logs' in st.session_state:
                st.session_state.logs.append({
                    'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3],  # Custom timestamp
                    'level': record.levelname,
                    'message': log_entry,
                    'agent': getattr(record, 'name', 'unknown')
                })
                # Keep only last 500 logs to prevent memory issues
                if len(st.session_state.logs) > 500:
                    st.session_state.logs = st.session_state.logs[-500:]
        except Exception as e:
            # Fallback in case of error
            print(f"Error capturing log: {e}")

# Setup logging to capture logs from agents
def setup_log_capture():
    # Get the root logger and add Streamlit handler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add Streamlit handler
    streamlit_handler = StreamlitLogHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    streamlit_handler.setFormatter(formatter)
    root_logger.addHandler(streamlit_handler)
    
    # Also add console handler for debugging (optional)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return root_logger

# Initialize logging
logger = setup_log_capture()

server_url = st.sidebar.text_input("MCP Server SSE URL", value="http://localhost:8000/sse")

# Sidebar configuration
with st.sidebar:
    st.subheader("⚙️ Configuration")
    show_logs = st.checkbox("Show Live Logs", value=True)
    log_level_filter = st.multiselect(
        "Log Level Filter",
        ["DEBUG", "INFO", "WARNING", "ERROR"],
        default=["INFO", "WARNING", "ERROR"]
    )
    auto_scroll = st.checkbox("Auto-scroll logs", value=True)
    
    if st.button("Clear Logs"):
        st.session_state.logs = []
        st.rerun()

with st.form("claim_form"):
    col1, col2 = st.columns(2)
    with col1:
        customer_name = st.text_input("Customer Name", value="Allison Hill")
        customer_id = st.text_input("Customer ID", value="CUST-93810")
    with col2:
        st.markdown("### Claim Details")
        claim_amount = st.number_input("Claim Amount (USD)", min_value=0.0, value=8000.0, step=1000.0)
    
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
    submitted = st.form_submit_button("🚀 Process Claim", type="primary")

async def process_claim(text, name, cid, amount):
    logger.info(f"Starting claim processing for customer: {name}, ID: {cid}")
    logger.debug(f"Claim text: {text[:200]}...")  # Log first 200 chars
    logger.debug(f"Claim amount: ${amount:,.2f}")
    
    try:
        logger.info(f"Connecting to MCP server at {server_url}")
        async with Client(server_url) as client:
            logger.debug("Connected to MCP server successfully")
            
            logger.info("Calling complete_pipeline tool")
            result = await client.call_tool(
                "complete_pipeline",
                {"claim_text": text, "customer_name": name, "customer_id": cid},
            )
            logger.debug("Received response from pipeline")
            
            raw = result.content[0].text
            logger.debug(f"Raw response length: {len(raw)} characters")
            
            # Parse the response
            clean = raw.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:-1])
                logger.debug("Removed markdown code blocks from response")
            
            parsed_result = json.loads(clean)
            logger.info("Successfully parsed JSON response")
            return parsed_result
    except Exception as e:
        logger.error(f"Error in process_claim: {type(e).__name__}: {str(e)}", exc_info=True)
        return {"error": f"{type(e).__name__}: {str(e)}"}

if submitted:
    # Clear previous logs for new submission
    st.session_state.logs = []
    
    with st.spinner("Processing claim..."):
        logger.info("=" * 60)
        logger.info("NEW CLAIM SUBMISSION")
        logger.info("=" * 60)
        
        result = asyncio.run(process_claim(claim_text, customer_name, customer_id, claim_amount))
        
        logger.info("Claim processing completed")

    # Create main layout with two columns if logs are shown
    if show_logs:
        col_main, col_logs = st.columns([2, 1])
        
        with col_main:
            # Display results in main column
            if "error" in result:
                st.error(f"⚠️ {result['error']}")
                logger.error(f"Claim processing failed: {result['error']}")
            else:
                # Display main decision
                final_decision = result.get("final_decision", "N/A")
                if final_decision == "AUTO_APPROVE":
                    st.success(f"✅ {final_decision}")
                    logger.info(f"Final decision: {final_decision} - Claim approved")
                elif final_decision in ["AUTO_DENY", "POLICY_NOT_FOUND"]:
                    st.error(f"❌ {final_decision}")
                    logger.warning(f"Final decision: {final_decision} - Claim denied")
                else:
                    st.warning(f"⚠️ {final_decision}")
                    logger.info(f"Final decision: {final_decision} - Claim requires review")

                col1, col2, col3 = st.columns(3)
                with col1:
                    risk_score = result.get('risk_score', 0)
                    st.metric("Risk Score", f"{risk_score:.3f}")
                    logger.debug(f"Risk score displayed: {risk_score:.3f}")
                with col2:
                    fraud_prob = result.get('fraud_probability', 0)
                    st.metric("Fraud Probability", f"{fraud_prob:.3f}")
                    logger.debug(f"Fraud probability displayed: {fraud_prob:.3f}")
                with col3:
                    status = result.get("status", "UNKNOWN")
                    st.metric("Status", status)
                    logger.debug(f"Claim status: {status}")

                # Show extracted data
                with st.expander("📄 Extracted Claim Data"):
                    extracted = result.get("extracted_data", {})
                    st.json(extracted)
                    logger.debug("Displayed extracted claim data")

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
                    logger.debug("Displayed policy data")

                # Show warnings and violations
                if result.get("risk_violations"):
                    st.error("🚨 Risk Violations")
                    for v in result["risk_violations"]:
                        st.write(f"- **{v.get('rule_name')}**: {v.get('message')}")
                        logger.warning(f"Risk violation: {v.get('rule_name')} - {v.get('message')}")

                if result.get("risk_warnings"):
                    st.warning("⚠️ Risk Warnings")
                    for w in result["risk_warnings"]:
                        st.write(f"- **{w.get('rule_name')}**: {w.get('message')}")
                        logger.warning(f"Risk warning: {w.get('rule_name')} - {w.get('message')}")

                if result.get("policy_warnings"):
                    st.info("📌 Policy Warnings")
                    for w in result["policy_warnings"]:
                        st.write(f"- {w}")
                        logger.info(f"Policy warning: {w}")

                # Final recommendation
                st.subheader("💡 Recommended Action")
                recommendation = result.get("recommended_action", "No recommendation")
                st.write(recommendation)
                logger.info(f"Recommended action: {recommendation}")

                with st.expander("🔍 Full JSON Response"):
                    st.json(result)
                    logger.debug("Displayed full JSON response")
        
        with col_logs:
            st.subheader("📋 Live Logs")
            
            # Agent filter
            if st.session_state.logs:
                agents = sorted(list(set([log['agent'] for log in st.session_state.logs])))
                agent_filter = st.multiselect(
                    "Filter by agent",
                    agents,
                    default=agents if agents else []
                )
            else:
                agent_filter = []
                st.info("No logs yet")
            
            # Display logs
            log_container = st.container(height=400)
            with log_container:
                filtered_logs = [
                    log for log in st.session_state.logs 
                    if log['level'] in log_level_filter and (not agent_filter or log['agent'] in agent_filter)
                ]
                
                if not filtered_logs:
                    st.info("No logs to display")
                else:
                    for log in filtered_logs[-100:]:  # Show last 100 logs
                        if log['level'] == 'ERROR':
                            st.error(f"`{log['timestamp']}` **{log['agent']}** - {log['message']}")
                        elif log['level'] == 'WARNING':
                            st.warning(f"`{log['timestamp']}` **{log['agent']}** - {log['message']}")
                        elif log['level'] == 'INFO':
                            st.info(f"`{log['timestamp']}` **{log['agent']}** - {log['message']}")
                        else:  # DEBUG
                            st.caption(f"`{log['timestamp']}` **{log['agent']}** - {log['message']}")
            
            # Export logs button
            if st.session_state.logs and st.button("📥 Export Logs"):
                log_export = json.dumps(st.session_state.logs, indent=2)
                st.download_button(
                    label="Download Logs",
                    data=log_export,
                    file_name=f"claim_logs_{customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    else:
        # Display results without logs column (same as before but without logs)
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
                st.metric("Risk Score", f"{result.get('risk_score', 0):.3f}")
            with col2:
                st.metric("Fraud Probability", f"{result.get('fraud_probability', 0):.3f}")
            with col3:
                st.metric("Status", result.get("status", "UNKNOWN"))

            # Show extracted data
            with st.expander("📄 Extracted Claim Data"):
                extracted = result.get("extracted_data", {})
                st.json(extracted)

            # Show policy data
            with st.expander("📋 Policy Data"):
                policy = result.get("policy_data", {})
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

# Add status bar at the bottom of sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")
st.sidebar.markdown(f"**Logs captured:** {len(st.session_state.logs)}")
if st.session_state.logs:
    st.sidebar.markdown(f"**Last log:** {st.session_state.logs[-1]['timestamp']}")