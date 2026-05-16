#!/usr/bin/env python3
"""
Streamlit GUI Client for MCP Insurance System
"""

import streamlit as st
import asyncio
import json
from pathlib import Path
import sys
from src.mcp_submission_parsing.clients.cli_client import MCPInsuranceClient


# Page config
st.set_page_config(
    page_title="Insurance Claims Processor",
    page_icon="📋",
    layout="wide"
)

# Title
st.title("📋 Insurance Claims Processing System")
st.markdown("---")


async def process_claim_async(claim_text: str, customer_name: str):
    """Async claim processing"""
    client = MCPInsuranceClient()
    return await client.process_complete_claim(claim_text, customer_name)


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    use_llm = st.checkbox("Use LLM Enhancement", value=True)
    
    st.header("📊 Statistics")
    if 'total_processed' not in st.session_state:
        st.session_state.total_processed = 0
        st.session_state.results = []
    
    st.metric("Claims Processed", st.session_state.total_processed)


# Main area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Claim Submission")
    
    # Example claims
    example = st.selectbox(
        "Load Example Claim",
        ["Custom", "Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft"]
    )
    
    if example == "Single Vehicle Collision":
        default_text = """I hit a deer on Highway 101 last night at 11:30 PM. 
        My 2020 Toyota Camry has significant front-end damage. 
        Policy number POL-123456. I'm John Smith from Springfield, IL.
        Police were called and a report was filed. Estimated damage $4,500."""
    elif example == "Multi-vehicle Collision":
        default_text = """Was involved in a 3-car pileup on I-95 at 5 PM yesterday.
        My 2019 Honda Civic was rear-ended. Policy POL-789012.
        Name: Sarah Johnson. Two witnesses saw everything.
        Police report available. Damage around $8,000."""
    elif example == "Vehicle Theft":
        default_text = """My 2022 Ford F-150 was stolen from the mall parking lot.
        Policy number POL-345678. Name: Mike Brown.
        Police report filed. Vehicle value $45,000."""
    else:
        default_text = ""
    
    claim_text = st.text_area(
        "Claim Description",
        value=default_text,
        height=200,
        placeholder="Enter claim details here..."
    )
    
    customer_name = st.text_input("Customer Name (optional)", placeholder="e.g., John Smith")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        parse_btn = st.button("🔍 Parse Only", use_container_width=True)
    with col_btn2:
        process_btn = st.button("⚡ Process Complete Claim", type="primary", use_container_width=True)
    with col_btn3:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_btn:
        st.rerun()


with col2:
    st.subheader("📄 Results")
    
    if process_btn and claim_text:
        with st.spinner("Processing claim through all agents..."):
            try:
                # Run async processing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    process_claim_async(claim_text, customer_name)
                )
                loop.close()
                
                if 'error' in result:
                    st.error(f"❌ Error: {result['error']}")
                else:
                    st.session_state.total_processed += 1
                    st.session_state.results.append(result)
                    
                    # Display results
                    status_color = {
                        "APPROVED": "🟢",
                        "DENIED": "🔴",
                        "UNDER_REVIEW": "🟡",
                        "WITH_SIU": "🟠"
                    }
                    
                    status_icon = status_color.get(result.get('status', ''), "⚪")
                    
                    st.metric(
                        "Final Decision",
                        f"{status_icon} {result.get('final_decision', 'UNKNOWN')}"
                    )
                    
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.metric(
                            "Risk Score",
                            f"{result.get('risk_assessment', {}).get('risk_score', 'N/A'):.3f}"
                        )
                    with col_metric2:
                        st.metric(
                            "Fraud Probability",
                            f"{result.get('fraud_prediction', {}).get('fraud_probability', 'N/A'):.3f}"
                        )
                    
                    # Expandable details
                    with st.expander("📋 Detailed Results"):
                        st.json(result)
                    
                    # Summary
                    st.success("✅ Claim processed successfully!")
                    st.info(result.get('summary', 'No summary available'))
                    
            except Exception as e:
                st.error(f"Error processing claim: {str(e)}")
    
    elif parse_btn and claim_text:
        with st.spinner("Parsing claim..."):
            try:
                client = MCPInsuranceClient()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    client.parse_claim(claim_text, use_llm)
                )
                loop.close()
                
                st.subheader("Extracted Fields")
                
                parsed = result.get('result', result)
                
                # Display in nice format
                for key, value in parsed.items():
                    if value and key not in ['extraction_confidence', 'missing_fields']:
                        st.text(f"**{key.replace('_', ' ').title()}:** {value}")
                
                if parsed.get('missing_fields'):
                    st.warning(f"Missing fields: {', '.join(parsed['missing_fields'])}")
                
                st.info(f"Extraction confidence: {parsed.get('extraction_confidence', 0):.1%}")
                
            except Exception as e:
                st.error(f"Error parsing claim: {str(e)}")


# History section
if st.session_state.results:
    st.markdown("---")
    st.subheader("📜 Processing History")
    
    for i, result in enumerate(reversed(st.session_state.results[-10:]), 1):
        with st.expander(f"Claim {i} - {result.get('final_decision', 'Unknown')}"):
            st.json(result)


# Footer
st.markdown("---")
st.caption("MCP Insurance Claims Processing System | Powered by AI Agents")