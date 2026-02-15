# 🌍 AI Travel Planner - Agentic Application

An intelligent travel planning assistant powered by LangChain and LangGraph that helps users plan comprehensive trips with real-time data, cost breakdowns, and detailed itineraries.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Workflow](#workflow)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Available Tools](#available-tools)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)
- [Contributing](#contributing)

---

## 🎯 Overview

This AI Travel Planner is an **agentic application** that leverages Large Language Models (LLMs) and multiple specialized tools to create comprehensive travel plans. It uses a **ReAct (Reasoning + Acting) pattern** with LangGraph to orchestrate intelligent decision-making and tool usage.

### What makes it "Agentic"?

The system autonomously:
- **Reasons** about user queries
- **Plans** which tools to use and in what order
- **Acts** by calling appropriate tools
- **Observes** results and adapts its approach
- **Iterates** until it has complete information

---

## 🏗️ Project Architecture

```
┌─────────────────┐
│   Streamlit UI  │  ← User Interface
└────────┬────────┘
         │ HTTP Request
         ↓
┌─────────────────┐
│  FastAPI Server │  ← Backend API
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   LangGraph     │  ← Agentic Workflow Engine
│   Agent Loop    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ↓         ↓
┌────────┐ ┌──────┐
│  LLM   │ │Tools │  ← Groq/OpenAI + Specialized Tools
└────────┘ └──────┘
    │         │
    └────┬────┘
         │
    ┌────┴────────────────────────────┐
    │                                 │
    ↓                                 ↓
┌─────────┐                    ┌──────────┐
│ Google  │                    │  Tavily  │
│ Places  │                    │  Search  │
└─────────┘                    └──────────┘
    │                                 │
    ↓                                 ↓
┌─────────┐                    ┌──────────┐
│ Weather │                    │Currency  │
│   API   │                    │Converter │
└─────────┘                    └──────────┘
```

---

## 🔄 Workflow

### High-Level Flow

```
User Query → FastAPI → LangGraph Agent → Tools → Response Generation → User
```

### Detailed Agentic Workflow

The application follows a **ReAct (Reasoning + Acting) loop**:

```
┌──────────────────────────────────────────────────────────────┐
│                      AGENTIC LOOP                            │
│                                                              │
│  1. RECEIVE QUERY                                           │
│     ↓                                                        │
│  2. REASON (LLM analyzes what information is needed)        │
│     ↓                                                        │
│  3. PLAN (Decide which tools to use)                        │
│     ↓                                                        │
│  4. ACT (Execute tool calls)                                │
│     │                                                        │
│     ├→ Search Attractions (Google Places/Tavily)            │
│     ├→ Search Restaurants                                   │
│     ├→ Search Activities                                    │
│     ├→ Search Transportation                                │
│     ├→ Get Weather Info                                     │
│     ├→ Calculate Costs                                      │
│     └→ Convert Currency                                     │
│     ↓                                                        │
│  5. OBSERVE (Analyze tool outputs)                          │
│     ↓                                                        │
│  6. DECIDE                                                  │
│     ├→ Need more info? → Go back to step 2                 │
│     └→ Have enough info? → Proceed to step 7               │
│     ↓                                                        │
│  7. SYNTHESIZE (Generate comprehensive travel plan)         │
│     ↓                                                        │
│  8. RETURN RESPONSE (Formatted markdown output)             │
└──────────────────────────────────────────────────────────────┘
```

### Step-by-Step Example

**User Input:** "Plan a 5-day trip to Goa"

1. **Initial Reasoning**
   - LLM analyzes: Need attractions, hotels, restaurants, weather, costs
   
2. **Tool Planning**
   - Agent decides: "I need to search attractions, restaurants, activities, weather"

3. **Tool Execution** (Parallel/Sequential)
   ```
   → search_attractions("Goa")
   → search_restaurants("Goa") 
   → search_activities("Goa")
   → get_weather_forecast("Goa")
   → search_transportation("Goa")
   ```

4. **Observation**
   - Agent receives: Beach names, restaurant lists, activity options, weather data

5. **Cost Calculation**
   ```
   → estimate_total_hotel_cost(price_per_night=3000, total_days=5)
   → calculate_total_expense(hotel=15000, food=10000, activities=8000)
   → calculate_daily_expense_budget(total=33000, days=5)
   ```

6. **Synthesis**
   - Agent creates structured itinerary with all gathered information

7. **Output**
   - Returns formatted Markdown response with complete travel plan

---

## ✨ Features

### Core Capabilities

- ✅ **Real-time Data**: Fetches live information from Google Places, Tavily, and weather APIs
- ✅ **Dual Itineraries**: Provides both tourist hotspots and off-beat locations
- ✅ **Cost Breakdown**: Detailed expense calculation with daily budgets
- ✅ **Weather Forecasting**: Current weather and multi-day forecasts
- ✅ **Multi-source Search**: Falls back to Tavily if Google Places fails
- ✅ **Currency Conversion**: Supports international currency conversions
- ✅ **Responsive UI**: Clean Streamlit interface with real-time updates

### Output Includes

- 📅 Day-by-day itinerary
- 🏨 Hotel recommendations with pricing
- 🎯 Tourist attractions with details
- 🍽️ Restaurant suggestions with price ranges
- 🎪 Activities and experiences
- 🚗 Transportation options
- 💰 Complete cost breakdown
- 💵 Per-day budget estimates
- 🌤️ Weather information

---

## 🛠️ Tech Stack

### Backend
- **FastAPI**: REST API server
- **LangChain**: LLM orchestration framework
- **LangGraph**: Agentic workflow engine
- **Groq/OpenAI**: LLM providers

### Frontend
- **Streamlit**: Interactive web UI

### APIs & Tools
- **Google Places API**: Location and business data
- **Tavily API**: Fallback search engine
- **OpenWeatherMap API**: Weather data
- **Alpha Vantage**: Currency exchange rates

### Infrastructure
- **Render**: Deployment platform
- **Python 3.9+**: Runtime environment

---

## 📁 Project Structure

```
ai-trip-planner/
│
├── agent/
│   └── agentic_workflow.py          # LangGraph agent definition
│
├── tools/
│   ├── place_search_tool.py         # Google Places + Tavily search
│   ├── weather_info_tool.py         # Weather forecasting
│   ├── expense_calculator_tool.py   # Cost calculations
│   ├── currency_conversion_tool.py  # Currency converter
│   └── arthmatic_op_tool.py         # Basic math operations
│
├── utils/
│   ├── place_info_search.py         # Google Places wrapper
│   ├── weather_info.py              # Weather API wrapper
│   ├── expense_calculator.py        # Calculator logic
│   ├── currency_converter.py        # Exchange rate logic
│   └── save_to_document.py          # Document generation
│
├── config/
│   └── config.yaml                  # LLM configuration
│
├── main.py                          # FastAPI application
├── streamlit_app.py                 # Streamlit frontend
├── prompt.py                        # System prompt for agent
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project metadata
├── setup.py                         # Package setup
└── README.md                        # Project documentation
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip or uv package manager
- API keys (see [Environment Variables](#environment-variables))

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ai-trip-planner.git
cd ai-trip-planner
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or using uv:

```bash
uv pip install -r requirements.txt
```

### Step 4: Install Package in Editable Mode

```bash
pip install -e .
```

---

## ⚙️ Configuration

### LLM Configuration

Edit `config/config.yaml` to choose your LLM provider:

```yaml
llm:
  openai:
    provider: "openai"
    model_name: "gpt-4"
  groq:
    provider: "groq"
    model_name: "llama-3.1-70b-versatile"
```

### System Prompt

Customize agent behavior in `prompt.py`:

```python
SYSTEM_PROMPT = SystemMessage(
    content="""You are a helpful AI Travel Agent...
    """
)
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
# LLM API Keys
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Place Search APIs
GPLACES_API_KEY=your_google_places_api_key
TAVILY_API_KEY=your_tavily_api_key

# Weather API
OPENWEATHERMAP_API_KEY=your_openweather_api_key

# Currency Conversion
EXCHANGE_RATE_API_KEY=your_exchange_rate_api_key
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
```

### How to Get API Keys

1. **Groq**: https://console.groq.com/keys
2. **OpenAI**: https://platform.openai.com/api-keys
3. **Google Places**: https://developers.google.com/maps/documentation/places/web-service/get-api-key
4. **Tavily**: https://tavily.com/
5. **OpenWeatherMap**: https://openweathermap.org/api
6. **Alpha Vantage**: https://www.alphavantage.co/support/#api-key

---

## 💻 Usage

### Running Locally

#### Option 1: Run Backend and Frontend Separately

**Terminal 1 - Start FastAPI Backend:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start Streamlit Frontend:**
```bash
streamlit run streamlit_app.py
```

#### Option 2: Run Backend Only (API Mode)

```bash
uvicorn main:app --reload
```

Access API docs at: `http://localhost:8000/docs`

### Using the Application

1. Open Streamlit UI (usually at `http://localhost:8501`)
2. Enter your travel query: "Plan a 5-day trip to Paris"
3. Click "Send"
4. Wait for the AI agent to process and gather information
5. Receive comprehensive travel plan in Markdown format

---

## 📡 API Endpoints

### POST `/query`

Submit a travel planning query.

**Request Body:**
```json
{
  "question": "Plan a 5-day trip to Tokyo"
}
```

**Response:**
```json
{
  "answer": "# 🌍 AI Travel Plan\n\n## Day 1: Tokyo Arrival...\n\n..."
}
```

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Plan a trip to Bali for 7 days"}'
```

---

## 🔧 Available Tools

The agent has access to the following tools:

### 1. Place Search Tools
- `search_attractions(place)` - Find tourist attractions
- `search_restaurants(place)` - Find dining options
- `search_activities(place)` - Find activities and experiences
- `search_transportation(place)` - Find transport options

### 2. Weather Tools
- `get_current_weather(city)` - Current weather conditions
- `get_weather_forecast(city)` - Multi-day forecast

### 3. Expense Tools
- `estimate_total_hotel_cost(price_per_night, total_days)` - Calculate hotel costs
- `calculate_total_expense(*costs)` - Sum all expenses
- `calculate_daily_expense_budget(total_cost, days)` - Calculate daily budget

### 4. Currency Tools
- `convert_currency(amount, from_currency, to_currency)` - Convert between currencies

### 5. Arithmetic Tools
- `multiply(a, b)` - Multiplication
- `add(a, b)` - Addition

---

## 🎨 Customization

### Adding New Tools

1. Create tool file in `tools/` directory:

```python
# tools/my_custom_tool.py
from langchain.tools import tool

class MyCustomTool:
    def __init__(self):
        self.my_tool_list = self._setup_tools()
    
    def _setup_tools(self):
        @tool
        def custom_function(input: str) -> str:
            """Description of what this tool does"""
            # Your logic here
            return result
        
        return [custom_function]
```

2. Register in `agentic_workflow.py`:

```python
from tools.my_custom_tool import MyCustomTool

# In GraphBuilder class
custom_tool = MyCustomTool()
tools.extend(custom_tool.my_tool_list)
```

### Modifying Agent Behavior

Edit `prompt.py` to change how the agent responds:

```python
SYSTEM_PROMPT = SystemMessage(
    content="""Your custom instructions here..."""
)
```

---

## 🌐 Deployment

### Deploying to Render

1. Create a `render.yaml`:

```yaml
services:
  - type: web
    name: ai-travel-planner
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: GROQ_API_KEY
        sync: false
      - key: GPLACES_API_KEY
        sync: false
      # ... add all env vars
```

2. Connect your GitHub repository to Render
3. Add environment variables in Render dashboard
4. Deploy!

### Deploying Streamlit

Update `streamlit_app.py` with your deployed backend URL:

```python
BASE_URL = "https://your-backend-url.onrender.com"
```

Deploy to Streamlit Cloud:
```bash
# Push to GitHub
git push origin main

# Deploy via Streamlit Cloud dashboard
# https://streamlit.io/cloud
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue: Google Places API returns no results**
- Solution: The app automatically falls back to Tavily search

**Issue: Currency conversion fails**
- Solution: Check ALPHAVANTAGE_API_KEY is valid and has quota remaining

**Issue: LangGraph visualization not working**
- Solution: Ensure graphviz is installed: `pip install pygraphviz`

**Issue: API timeout errors**
- Solution: Increase timeout in `main.py` or use faster LLM model

---

## 📊 Performance Metrics

- **Average Response Time**: 10-30 seconds (depending on LLM and tool calls)
- **Tool Calls per Query**: 5-12 (varies by query complexity)
- **API Success Rate**: 95%+ (with fallback mechanisms)

---

## 🔮 Future Enhancements

- [ ] Add flight booking integration
- [ ] Support multi-city itineraries
- [ ] Add user preferences and memory
- [ ] Implement PDF export for itineraries
- [ ] Add real-time booking capabilities
- [ ] Support for group travel planning
- [ ] Integration with calendar apps
- [ ] Mobile app version

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Authors

- **Mayank Pathak** - *Initial work* - [mayank74pathak@gmail.com](mailto:mayank74pathak@gmail.com)
- **Atriyo** - *Travel Agent Creator*

---

## 🙏 Acknowledgments

- LangChain for the amazing framework
- LangGraph for agentic workflow capabilities
- Google Places API for location data
- Tavily for search fallback
- OpenWeatherMap for weather data
- Groq for fast LLM inference

---

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: mayank74pathak@gmail.com

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

---

**Made with ❤️ by the AI Travel Planner Team** 
