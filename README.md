# MusicStreamAnalysis
Music Streaming Analytics is an interactive, data-driven Streamlit application designed to analyze, visualize, and explore music streaming behavior across multiple dimensions such as songs, artists, genres, demographics, devices, and time trends. Designed with rich visuals, flexible search tools, ML integration placeholders, and full data management support.
## ⭐ Features:
## 📊 Interactive Dashboard
- Total streams, unique songs, unique artists
- Average song duration
- Streams by genre and country
- Daily streaming time-series trends

## 🔍 Search & Explore
- Search and analyze data by:
- Song
- Artist
- Genre
Includes:
✅ Summary metrics
✅ Detailed aggregated results table
📈 3. Popular Content Insights
- Top 10 most-streamed songs
- Top 10 most-streamed artists
- Trending now (last 7 days)

## ⏰ Temporal Analysis
- Streams by time of day
- Streams by day of week
- Monthly streaming trends

## 👥 Demographic Analysis
- Streams broken down by age group
- Country-wise distribution
- Genre preference across age demographics

## 🎯 Behavioral Metrics
- Device usage distribution
- Average listening duration per device
- Sunburst chart of listening patterns by device & time of day

✅Engagement metrics:
- Avg daily streams
- Peak listening time
- Most used device

## 🤖 ML Integration (Demo Mode)
- Supports selection of ML providers:
- OpenAI
- Anthropic Claude
- Google PaLM
- Hugging Face

✅Features:
- API key input + connection test
- Simulated model predictions
- AI insights (trends, anomalies, segmentation)

## MongoDB connection and operations
- connect to localhost either local or Atlas
- perform any operation
- Uses document-based storage for flexible music streaming records

## Data Management
- View current dataset
- Generate sample data
- Reset to default data
- Export dataset as CSV

## Technology Stack
- Python
- Streamlit
- Pandas
- MongoDB

## Installation
pip install streamlit pandas numpy plotly

## Run the App
streamlit run main.py

## connect to mongoDB
start mongo server 
copy the connection string and paste
connect

## 📁 Project Structure
.
├── main.py
├── README.md
└── data/ (optional)

## Summary
- Music Streaming Analytics offers a complete, interactive environment for exploring and understanding streaming data with:
- Multi-dimensional filtering
- ML-ready structure
- Perfect for analytics projects, dashboards, academic work, or music industry insights.
