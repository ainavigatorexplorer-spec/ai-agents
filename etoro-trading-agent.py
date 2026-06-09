import os
import json
import uuid
import asyncio
from typing import Any, Dict, List
import httpx
from groq import Groq

# =====================================================================
# 1. LIVE CONFIGURATION PROFILE
# =====================================================================
CONFIG = {
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
    "GROQ_MODEL": "llama3-70b-8192",  # Elite logic reasoning for multi-step execution
    
    # Account Environment Selection: Choose "virtual" or "real"
    "ACCOUNT_TYPE": "virtual", 
    
    # Credentials retrieved from eToro Platform (Settings > Trading)
    "ETORO": {
        "BASE_URL": "https://public-api.etoro.com",
        "API_KEY": os.getenv("ETORO_API_KEY", "your_public_api_key_here"),
        "USER_KEY": os.getenv("ETORO_USER_KEY", "your_user_specific_key_here"),
    },
    
    # Risk Guardrails & Trade Formulation Rules
    "TRADING_PREFERENCES": {
        "max_risk_per_trade_pct": 1.5,
        "default_leverage": 5,
        "stop_loss_pct": 2.0,
        "take_profit_pct": 6.0
    }
}

# =====================================================================
# 2. IMPLEMENTING ETORO ENGINE NATIVELY MATCHING SPECIFICATIONS
# =====================================================================
class EToroClient:
    """Natively executes requests based on https://api-portal.etoro.com/ specifications."""
    
    def __init__(self):
        self.base_url = CONFIG["ETORO"]["BASE_URL"]
        self.headers_template = {
            "x-api-key": CONFIG["ETORO"]["API_KEY"],
            "x-user-key": CONFIG["ETORO"]["USER_KEY"],
            "Content-Type": "application/json"
        }

    def _get_headers(self) -> Dict[str, str]:
        """Injects a unique request UUID header (x-request-id) into every outbound call."""
        headers = self.headers_template.copy()
        headers["x-request-id"] = str(uuid.uuid4())
        return headers

    async def resolve_symbol_to_id(self, symbol: str) -> Dict[str, Any]:
        """Resolves raw text tickers to immutable numeric eToro Instrument IDs."""
        url = f"{self.base_url}/api/v1/market-data/search"
        params = {"internalSymbolFull": symbol.upper()}
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=self._get_headers(), params=params, timeout=10.0)
                return response.json()
            except Exception as e:
                return {"error": f"Failed to resolve symbol {symbol}: {str(e)}"}

    async def get_market_rates(self) -> Dict[str, Any]:
        """Fetches active market pricing and execution rates for discovery analysis."""
        url = f"{self.base_url}/api/v1/market-data/instruments/rates"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=self._get_headers(), timeout=10.0)
                return response.json()
            except Exception as e:
                return {"error": f"Failed to fetch market rates: {str(e)}"}

    async def get_portfolio_pnl(self, environment: str) -> Dict[str, Any]:
        """Retrieves targeted portfolio details, credit buffers, and unrealized positions."""
        env_path = "demo" if environment.lower() == "virtual" else "real"
        url = f"{self.base_url}/api/v1/trading/info/{env_path}/pnl"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=self._get_headers(), timeout=10.0)
                return response.json()
            except Exception as e:
                return {"error": f"Failed to pull PnL framework: {str(e)}"}

    async def open_market_order(self, environment: str, instrument_id: int, amount: float, leverage: int) -> Dict[str, Any]:
        """Executes a market order allocation by cash budget using V2 endpoints."""
        env_path = "demo/" if environment.lower() == "virtual" else ""
        url = f"{self.base_url}/api/v2/trading/execution/{env_path}orders"
        
        payload = {
            "InstrumentID": int(instrument_id),
            "Amount": float(amount),
            "Leverage": int(leverage),
            "StopLossPct": float(CONFIG["TRADING_PREFERENCES"]["stop_loss_pct"]),
            "TakeProfitPct": float(CONFIG["TRADING_PREFERENCES"]["take_profit_pct"])
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=self._get_headers(), json=payload, timeout=10.0)
                return response.json()
            except Exception as e:
                return {"error": f"Execution failed: {str(e)}"}


# =====================================================================
# 3. CONVERTER INTERFACES FOR MODEL CONTEXT PROTOCOL (MCP) MAPPING
# =====================================================================
etoro_service = EToroClient()

# Explicit Tool Metadata Schema corresponding to eToro SDK definitions
MCP_TOOLS_SPECIFICATION = [
    {
        "type": "function",
        "function": {
            "name": "resolve_symbol_to_id",
            "description": "Resolves a text ticker symbol (e.g. 'AAPL' or 'BTC') to its immutable, numeric eToro instrumentId.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "The exact ticker asset symbol"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_rates",
            "description": "Retrieves real-time market rates, conversion spreads, and active asset asset class pricing logs.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_pnl",
            "description": "Fetches systemic account configuration balances, equity variables, credit, and running active allocations.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_market_order",
            "description": "Places an absolute live or virtual trading deployment position by cash valuation size using system parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instrument_id": {"type": "integer", "description": "The numeric target ID derived from resolution lookup"},
                    "amount": {"type": "number", "description": "USD value threshold allocated for deployment"},
                    "leverage": {"type": "integer", "description": "Leverage multiplier tracking parameters"}
                },
                "required": ["instrument_id", "amount", "leverage"]
            }
        }
    }
]

async def call_mcp_tool(name: str, arguments: dict) -> str:
    """Router executing the explicit programmatic bindings mapped to the endpoints."""
    if name == "resolve_symbol_to_id":
        res = await etoro_service.resolve_symbol_to_id(arguments.get("symbol"))
    elif name == "get_market_rates":
        res = await etoro_service.get_market_rates()
    elif name == "get_portfolio_pnl":
        res = await etoro_service.get_portfolio_pnl(CONFIG["ACCOUNT_TYPE"])
    elif name == "open_market_order":
        res = await etoro_service.open_market_order(
            environment=CONFIG["ACCOUNT_TYPE"],
            instrument_id=arguments.get("instrument_id"),
            amount=arguments.get("amount"),
            leverage=arguments.get("leverage")
        )
    else:
        res = {"error": f"Tool execution method path '{name}' untracked."}
    return json.dumps(res)


# =====================================================================
# 4. SYSTEM INSTRUCTIONS AND PROCESSING RUNTIME LOOP
# =====================================================================
def get_orchestration_prompt() -> str:
    return f"""
You are a Day Trading AI Orchestrator running over the verified eToro Portal Public API Protocol schemas.
Operational Context: Operating via the {CONFIG['ACCOUNT_TYPE'].upper()} account engine infrastructure.

Mandatory Directives:
1. SEPARATION OF ID RESOLUTION: Never guess or output raw string assets into executions. You MUST resolve strings into strict numerical `instrumentId` properties using 'resolve_symbol_to_id' first.
2. DISCOVERY METRICS: If tracking market health optimization looks for extreme short term spread volatility and dense volumes, get active data arrays via 'get_market_rates'.
3. BALANCES & CAPITAL SAFETY: Before generating trade execution parameters, analyze credit profiles from 'get_portfolio_pnl'.
4. RISK CONTROL LIMITATIONS: Restrict individual positions based on maximum entry caps of {CONFIG['TRADING_PREFERENCES']['max_risk_per_trade_pct']}% of total portfolio balance metrics. Default to {CONFIG['TRADING_PREFERENCES']['default_leverage']}x leverage constraints unless requested otherwise.
"""

async def run_trading_agent(prompt_query: str):
    if not CONFIG["GROQ_API_KEY"]:
        raise ValueError("System execution failed: GROQ_API_KEY environment configuration value is missing.")
        
    client = Groq(api_key=CONFIG["GROQ_API_KEY"])
    
    messages = [
        {"role": "system", "content": get_orchestration_prompt()},
        {"role": "user", "content": prompt_query}
    ]
    
    print(f"📡 [Init] Engaging Core Agent. Mode: {CONFIG['ACCOUNT_TYPE'].upper()} Mode Pipeline.")
    
    for cycle in range(5):
        response = client.chat.completions.create(
            model=CONFIG["GROQ_MODEL"],
            messages=messages,
            tools=MCP_TOOLS_SPECIFICATION,
            tool_choice="auto",
            temperature=0.1
        )
        
        response_msg = response.choices[0].message
        messages.append(response_msg)
        
        if response_msg.tool_calls:
            for tool_call in response_msg.tool_calls:
                t_name = tool_call.function.name
                t_args = json.loads(tool_call.function.arguments)
                
                print(f"⚙️ [Executing Tool via Portal Spec] Call: '{t_name}' Parameters: {t_args}")
                tool_output = await call_mcp_tool(t_name, t_args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": t_name,
                    "content": tool_output
                })
        else:
            print("\n" + "="*70)
            print("📈 AGENT PORTAL REAL-TIME ANALYSIS LOG REPORT")
            print("="*70)
            print(response_msg.content)
            break

# =====================================================================
# 5. USER INTERFACE ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    print("✨ eToro Dev-Portal Verified Python AI Day Trader ✨")
    print("-------------------------------------------------------")
    print("1: Scan live tracking instruments for optimal market volatility positions.")
    print("2: Enter specific target symbol asset ticker for comprehensive pre-flight trade logic checks.")
    
    user_selection = input("\nSelect execution script choice (1 or 2): ").strip()
    
    if user_selection == "1":
        user_query = "Query system rates, extract top active options for rapid day trade positioning, check current portfolio constraints and draft safe risk parameters."
    elif user_selection == "2":
        tgt_symbol = input("Input target asset ticker identifier symbol (e.g., TSLA, BTC, ETH, AMZN): ").strip().upper()
        user_query = f"Check and resolve ticker '{tgt_symbol}' to its verified ID. Evaluate account equity metrics, check real time rates, and execute an optimal allocation strategy under risk boundaries."
    else:
        print("Incorrect value selected. Ending engine cycle.")
        exit()

    asyncio.run(run_trading_agent(user_query))
