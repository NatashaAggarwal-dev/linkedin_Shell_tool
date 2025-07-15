import gradio as gr
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults  # ✅ New tool

# 1. Set up Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="AIzaSyCoVI0AUkemJSo51ZbhveoWhy94u-otqus",  # 🔐 Replace with your real Gemini key
    temperature=0.7
)

# 2. Use DuckDuckGo Search Tool instead of placeholder
search_tool = DuckDuckGoSearchResults()
tools = [search_tool]

# 3. Create the LLM agent
agent = create_react_agent(llm, tools)

# 4. Define the main function for LinkedIn post generation
def generate_carousel(project_topic):
    input_message = {
        "role": "user",
        "content": (
            f"You are a professional LinkedIn content writer. "
            f"Create a carousel post for this project:\n"
            f"{project_topic}\n\n"
            f"Include:\n"
            f"- Slide 1: Title and subtitle\n"
            f"- Slide 2: Features or how it works\n"
            f"- Slide 3: Tools used / learning outcomes\n"
            f"- Slide 4: Call-to-action + GitHub/project link\n\n"
            f"Also provide a LinkedIn caption with hashtags and emojis.\n\n"
            f"(⚙ Optional: Suggest how this can be repurposed for blog posts, tweets, or slide images.)"
        ),
    }

    full_output = ""
    for step in agent.stream({"messages": [input_message]}, stream_mode="values"):
        full_output += step["messages"][-1].content + "\n"

    return full_output

# 5. Gradio Interface
iface = gr.Interface(
    fn=generate_carousel,
    inputs=gr.Textbox(
        label="Enter your project topic",
        placeholder="e.g. WhatsApp automation using Python and pywhatkit"
    ),
    outputs=gr.Textbox(
        label="Generated LinkedIn Carousel + Caption",
        lines=25
    ),
    title="📣 LinkedIn Post Generator (Gemini LLM)",
    description=(
        "Generate LinkedIn carousel content + caption using Gemini LLM. 💡\n\n"
        "⚙ Easy to extend for blogs, tweets, or GitHub READMEs.\n"
        "🖼 Future scope: Connect to Canva API or PIL for slide generation."
    ),
    theme="default"
)

# 6. Run the App
iface.launch()