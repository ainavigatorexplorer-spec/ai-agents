import os
import json
import requests
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Initialize environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEED_MCP_URL = os.getenv("INDEED_MCP_URL", "https://mcp.indeed.com/claude/mcp")
INDEED_MCP_TOKEN = os.getenv("INDEED_MCP_TOKEN")

if not GROQ_API_KEY:
    raise ValueError("Error: Please set your GROQ_API_KEY in the environment or a .env file.")

# Initialize the Groq SDK client
client = Groq(api_key=GROQ_API_KEY)

# Define Indeed MCP tools utilizing standard JSON Schema layout for Groq tool-use
INDEED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "job_search",
            "description": "Search Indeed jobs by title, keywords, location, and employment type. Returns job listings with titles, companies, locations, salaries, and application URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Job titles, roles, or specific skills to look for (e.g., 'Python Developer')."},
                    "location": {"type": "string", "description": "Geographical location or specific terms like 'remote'."},
                    "employment_type": {
                        "type": "string", 
                        "enum": ["fulltime", "parttime", "contract", "internship", "temporary"],
                        "description": "The specific arrangement or commitment structure."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "job_detail",
            "description": "Get descriptive, in-depth information for an Indeed job listing using its unique Job ID. Includes full lists of requirements, qualifications, and benefits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "The unique identifying string of the specific job posting."}
                },
                "required": ["job_id"]
            }
        }
    }
]

def call_indeed_mcp(tool_name: str, arguments: dict) -> dict:
    """
    Executes a protocol request to the Indeed MCP server using the standardized 
    Model Context Protocol format over HTTP JSON-RPC. Falls back to a high-fidelity 
    simulation if live credentials are not present.
    """
    # Fallback simulation if an active Bearer Token isn't provided
    if not INDEED_MCP_TOKEN:
        print(f"\n[🔄 MCP Client -> Simulating live Indeed MCP Server tool execution for: '{tool_name}']")
        if tool_name == "job_search":
            kw = arguments.get("keywords", "Software Engineer")
            loc = arguments.get("location", "Remote")
            return {
                "jobs": [
                    {
                        "job_id": "job_idx_991",
                        "title": f"Senior {kw}",
                        "company": "Apex Nexus Analytics",
                        "location": loc,
                        "salary": "$135,000 - $160,000 / year",
                        "employment_type": "fulltime",
                        "url": "https://www.indeed.com/viewjob?jk=job_idx_991"
                    },
                    {
                        "job_id": "job_idx_992",
                        "title": f"Staff {kw} - Systems Specialist",
                        "company": "Quantum Core Labs",
                        "location": "Hybrid",
                        "salary": "$110,000 - $130,000 / year",
                        "employment_type": "fulltime",
                        "url": "https://www.indeed.com/viewjob?jk=job_idx_992"
                    }
                ]
            }
        elif tool_name == "job_detail":
            return {
                "job_id": arguments.get("job_id"),
                "description": "Seeking an expert engineer capable of deploying scalable workflows, utilizing Python, and managing complex API frameworks.",
                "requirements": ["Python", "REST/GraphQL APIs", "Distributed Architecture", "Docker"],
                "benefits": ["Remote Workspace Allowance", "Comprehensive Premium Health Plan", "Generous PTO"]
            }
        return {"error": f"Tool '{tool_name}' not recognized."}

    # Live connection executing the standard JSON-RPC layout for remote MCP servers
    headers = {
        "Authorization": f"Bearer {INDEED_MCP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "jsonrpc": "2.0",
        "id": "groq-mcp-agent-session",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    
    try:
        response = requests.post(INDEED_MCP_URL, json=payload, headers=headers)
        response.raise_for_status()
        rpc_response = response.json()
        return rpc_response.get("result", rpc_response)
    except Exception as e:
        return {"error": f"Failed to successfully communicate with Indeed MCP: {str(e)}"}

def run_job_matching_agent():
    print("=" * 60)
    print("      🎯 WELCOME TO YOUR AI JOB MATCHING & PROFILE AGENT 🎯")
    print("  Provide your preferences below. Press [Enter] to leave optional fields blank.")
    print("=" * 60)
    
    # Optional User Prompts
    job_role = input("💼 Target Job Role / Title : ").strip()
    skills   = input("🛠️  Key Skills / Keywords  : ").strip()
    job_type = input("🏡 Job Type (remote/hybrid/onsite): ").strip().lower()
    salary   = input("💰 Expected Salary Range    : ").strip()
    
    # Construct structured matching context for the LLM
    criteria_list = []
    if job_role: criteria_list.append(f"- Job Roles: {job_role}")
    if skills:   criteria_list.append(f"- Candidate Skills: {skills}")
    if job_type: criteria_list.append(f"- Location Configuration: {job_type}")
    if salary:   criteria_list.append(f"- Remuneration Expectation: {salary}")
    
    formatted_criteria = "\n".join(criteria_list) if criteria_list else "- No constraints applied. Query generally available positions."

    # Define behavior boundaries for the agent
    system_instructions = (
        "You are an expert AI Recruitment Specialist configured to connect directly to the Indeed platform via MCP tools. "
        "Your core task is to evaluate user criteria and call 'job_search' to find opportunities. "
        "Analyze the job profiles, salaries, matching skills, and workplace types. "
        "Highlight exactly where listings perfectly match or deviate from the user's optional input metrics."
    )
    
    user_request = f"Please fetch and synthesize matching profiles using these criteria:\n\n{formatted_criteria}"
    
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_request}
    ]
    
    print("\n🤖 Groq AI Agent is planning search strategy and connecting to Indeed MCP...")
    
    # First LLM Call: Give the model the tools and let it generate parameters
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",  # Fast, highly accurate tool-use capability
        messages=messages,
        tools=INDEED_TOOLS,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    # Process tool requests loop if the model decides to query the server
    if tool_calls:
        messages.append(response_message)
        
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            
            # Fire tool against our MCP interface
            mcp_output = call_indeed_mcp(func_name, func_args)
            
            # Feed the protocol context back to the chat history
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": json.dumps(mcp_output)
            })
        
        print("📊 Analyzing returned telemetry and formatting job profile summaries...")
        
        # Second LLM Call: Generate final synthesis after consuming data
        final_response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages
        )
        
        print("\n" + "=" * 55)
        print("✨ AGENT PROFILE INTELLIGENCE REPORT ✨")
        print("=" * 55)
        print(final_response.choices[0].message.content)
        
    else:
        # Fallback if the agent processes without requiring tool lookup
        print("\n" + "=" * 55)
        print("✨ AGENT ASSISTANCE REPORT ✨")
        print("=" * 55)
        print(response_message.content)

if __name__ == "__main__":
    run_job_matching_agent()
