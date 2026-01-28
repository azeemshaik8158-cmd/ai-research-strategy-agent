import os

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# --- Environment & configuration ---
OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def check_env_vars():
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    return missing

@st.cache_resource
def get_agent():
    """Create and cache the LangChain agent."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    tavily = TavilySearchResults(max_results=3)

    # websearch tool
    @tool
    def search(query: str) -> str:
        """
        Perform a real-time web search using Tavily
        to retrieve the latest AI industry news.
        """
        return str(tavily.invoke(query))

    # Analysis tool
    @tool
    def analyze(text: str) -> str:
        """
        Analyze retrieved information to identify
        trends, risks, and opportunities.
        """
        return f"""
        You are an AI industry analyst.

        From the following data, extract:
        - Key trends
        - Potential risks
        - Emerging opportunities

        Data:
        {text}
        """

 # Recommendations tool
    @tool
    def recommend(analysis: str) -> str:
        """
        Generate actionable business recommendations
        based on the analysis.
        """
        return f"""
        You are advising a startup founder.

        Based on the analysis, provide:
        - Business impact
        - Clear, actionable recommendations

        Analysis:
        {analysis}
        """

    tools = [search, analyze, recommend]
    agent = create_agent(tools=tools, model=llm)
    return agent


def main():
    st.set_page_config(
        page_title="AI Research & Decision Assistant",
        page_icon="🤖",
        layout="wide",
    )

    st.title("AI Research & Decision Assistant")
    st.markdown(
        "Ask about **current AI industry news, trends, risks, and opportunities**.\n\n"
        "The app will search the web, analyze the information, and give you **actionable recommendations** "
        "for startup decisions."
    )

    with st.expander("Environment & status", expanded=False):
        missing = check_env_vars()
        if missing:
            st.error(
                "Missing environment variables:\n\n"
                + "\n".join(f"- `{m}`" for m in missing)
            )
            st.markdown(
                "Set these as environment variables locally, or as **Secrets** "
                "in Streamlit Community Cloud."
            )
        else:
            st.success("All required environment variables are set.")

    user_query = st.text_area(
        "Your question about the AI industry",
        value="Analyze current AI industry news and recommend actions for a startup.",
        height=150,
    )

    if st.button("Analyze & Recommend", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a question or topic first.")
            return

        missing = check_env_vars()
        if "OPENAI_API_KEY" in missing or "TAVILY_API_KEY" in missing:
            st.error(
                "You must set `OPENAI_API_KEY` and `TAVILY_API_KEY` before running the analysis."
            )
            return

        with st.spinner("Thinking, searching the web, and analyzing..."):
            agent = get_agent()
            try:
                response = agent.invoke(
                    {
                        "messages": [
                            ("user", user_query),
                        ]
                    }
                )
            except Exception as e:
                st.error(f"Error while running the agent: {e}")
                return

        # Try to extract the final message content
        final_text = None
        if isinstance(response, dict) and "messages" in response:
            try:
                final_text = response["messages"][-1].content
            except Exception:
                pass

        if final_text is None:
            final_text = str(response)

        st.subheader("Recommendations")
        st.markdown(final_text)


if __name__ == "__main__":
    main()


    
