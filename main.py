import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Music Streaming Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for pastel background and dark fonts
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #ffeef8 0%, #e6f3ff 50%, #fff5e6 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fce4ec 0%, #e1f5fe 50%, #fff9c4 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #1a252f !important;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: #1a252f !important;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #1a252f !important;
    }
    
    /* Main content */
    .main {
        color: #2c3e50;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1a252f !important;
        font-weight: 700;
    }
    
    .stMetric label {
        color: #34495e !important;
        font-weight: 600;
    }
    
    .stMetric value {
        color: #1a252f !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #1a252f;
    }
    
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: #34495e !important;
        font-weight: 600;
    }
    
    /* Force black text for all paragraphs, labels, spans, and list items */
    p, label, span, li, div[class*="stMarkdown"] {
        color: #000000 !important;
    }
    
    /* Force black text for markdown content */
    .stMarkdown, .stMarkdown * {
        color: #000000 !important;
    }
    
    /* Button text */
    button {
        color: #ffffff !important;
    }
    
    /* Text input labels */
    .stTextInput label, .stFileUploader label {
        color: #000000 !important;
        font-weight: 600;
    }
    
    /* Info, warning, success boxes */
    .stAlert {
        color: #000000 !important;
    }
    
    /* Radio button labels */
    .stRadio label {
        color: #000000 !important;
    }
    
    /* Selectbox text */
    .stSelectbox label {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for data
if 'streaming_data' not in st.session_state:
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-11-13', freq='D')
    
    songs = ['Blinding Lights', 'Shape of You', 'Someone Like You', 'Levitating', 
             'Watermelon Sugar', 'Bad Guy', 'Circles', 'Señorita', 'Sunflower', 'Rockstar']
    artists = ['The Weeknd', 'Ed Sheeran', 'Adele', 'Dua Lipa', 
               'Harry Styles', 'Billie Eilish', 'Post Malone', 'Shawn Mendes', 'Post Malone', 'Post Malone']
    genres = ['Pop', 'Pop', 'Pop', 'Pop', 'Pop', 'Alternative', 'Hip-Hop', 'Pop', 'Hip-Hop', 'Hip-Hop']
    
    data = []
    for _ in range(5000):
        idx = np.random.randint(0, len(songs))
        data.append({
            'date': np.random.choice(dates),
            'song': songs[idx],
            'artist': artists[idx],
            'genre': genres[idx],
            'streams': np.random.randint(1000, 50000),
            'duration_sec': np.random.randint(180, 300),
            'age_group': np.random.choice(['13-17', '18-24', '25-34', '35-44', '45+']),
            'country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'Germany', 'India']),
            'device': np.random.choice(['Mobile', 'Desktop', 'Tablet', 'Smart Speaker']),
            'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'])
        })
    
    st.session_state.streaming_data = pd.DataFrame(data)

# Sidebar
st.sidebar.title("🎵 Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Dashboard",
    "🔍 Search & Explore",
    "📈 Popular Content Insights",
    "⏰ Temporal Analysis",
    "👥 Demographic Analysis",
    "🎯 Behavioral Metrics",
    "🤖 ML Integration",
    "📤 Upload CSV Data",
    "📥 Data Management"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
df = st.session_state.streaming_data.copy()

# Date range filter
min_date = df['date'].min().date()
max_date = df['date'].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply date filter
if len(date_range) == 2:
    df = df[(df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])]

# Genre filter
genres = st.sidebar.multiselect("Genre", options=df['genre'].unique().tolist(), default=df['genre'].unique().tolist())
if genres:
    df = df[df['genre'].isin(genres)]

# ==================== DASHBOARD PAGE ====================
if page == "📊 Dashboard":
    st.title("🎵 Music Streaming Analytics Dashboard")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Streams", f"{df['streams'].sum():,}")
    with col2:
        st.metric("Unique Songs", df['song'].nunique())
    with col3:
        st.metric("Unique Artists", df['artist'].nunique())
    with col4:
        avg_duration = df['duration_sec'].mean()
        st.metric("Avg Duration", f"{int(avg_duration//60)}:{int(avg_duration%60):02d}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Streams by Genre")
        genre_streams = df.groupby('genre')['streams'].sum().reset_index()
        fig = px.pie(genre_streams, values='streams', names='genre', 
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Streams by Country")
        country_streams = df.groupby('country')['streams'].sum().reset_index()
        fig = px.bar(country_streams, x='country', y='streams',
                     color='streams', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series
    st.subheader("📈 Daily Streaming Trends")
    daily_streams = df.groupby('date')['streams'].sum().reset_index()
    fig = px.line(daily_streams, x='date', y='streams', 
                  labels={'streams': 'Total Streams', 'date': 'Date'})
    fig.update_traces(line_color='#ff6b9d')
    st.plotly_chart(fig, use_container_width=True)

# ==================== SEARCH & EXPLORE PAGE ====================
elif page == "🔍 Search & Explore":
    st.title("🔍 Search & Explore")
    
    search_type = st.radio("Search by:", ["Song", "Artist", "Genre"])
    
    if search_type == "Song":
        search_term = st.text_input("Search for a song:", "")
        if search_term:
            results = df[df['song'].str.contains(search_term, case=False, na=False)]
        else:
            results = df
    elif search_type == "Artist":
        search_term = st.text_input("Search for an artist:", "")
        if search_term:
            results = df[df['artist'].str.contains(search_term, case=False, na=False)]
        else:
            results = df
    else:
        search_term = st.selectbox("Select Genre:", df['genre'].unique())
        results = df[df['genre'] == search_term]
    
    st.subheader(f"Found {len(results)} results")
    
    # Display summary
    if len(results) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Streams", f"{results['streams'].sum():,}")
        with col2:
            st.metric("Avg Streams", f"{results['streams'].mean():.0f}")
        with col3:
            st.metric("Unique Songs", results['song'].nunique())
        
        # Detailed table
        st.subheader("Detailed Results")
        display_df = results.groupby(['song', 'artist', 'genre']).agg({
            'streams': 'sum',
            'duration_sec': 'mean'
        }).reset_index()
        display_df = display_df.sort_values('streams', ascending=False)
        display_df['duration'] = display_df['duration_sec'].apply(
            lambda x: f"{int(x//60)}:{int(x%60):02d}"
        )
        st.dataframe(
            display_df[['song', 'artist', 'genre', 'streams', 'duration']].head(20),
            use_container_width=True
        )

# ==================== POPULAR CONTENT INSIGHTS ====================
elif page == "📈 Popular Content Insights":
    st.title("📈 Popular Content Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎵 Top 10 Songs")
        top_songs = df.groupby('song')['streams'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(top_songs, x=top_songs.values, y=top_songs.index, orientation='h',
                     labels={'x': 'Total Streams', 'y': 'Song'},
                     color=top_songs.values, color_continuous_scale='Purples')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎤 Top 10 Artists")
        top_artists = df.groupby('artist')['streams'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(top_artists, x=top_artists.values, y=top_artists.index, orientation='h',
                     labels={'x': 'Total Streams', 'y': 'Artist'},
                     color=top_artists.values, color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🔥 Trending Now")
    # Last 7 days trending
    recent_date = df['date'].max()
    week_ago = recent_date - timedelta(days=7)
    recent_df = df[df['date'] > week_ago]
    
    trending = recent_df.groupby('song')['streams'].sum().sort_values(ascending=False).head(5)
    
    for idx, (song, streams) in enumerate(trending.items(), 1):
        artist = df[df['song'] == song]['artist'].iloc[0]
        st.markdown(f"**{idx}. {song}** by *{artist}* - {streams:,} streams")

# ==================== TEMPORAL ANALYSIS ====================
elif page == "⏰ Temporal Analysis":
    st.title("⏰ Temporal Analysis")
    
    # Time of day analysis
    st.subheader("📊 Streams by Time of Day")
    time_streams = df.groupby('time_of_day')['streams'].sum().reset_index()
    time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    time_streams['time_of_day'] = pd.Categorical(time_streams['time_of_day'], 
                                                   categories=time_order, ordered=True)
    time_streams = time_streams.sort_values('time_of_day')
    
    fig = px.bar(time_streams, x='time_of_day', y='streams',
                 color='streams', color_continuous_scale='Sunset')
    st.plotly_chart(fig, use_container_width=True)
    
    # Day of week analysis
    st.subheader("📅 Streams by Day of Week")
    df['day_of_week'] = df['date'].dt.day_name()
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_streams = df.groupby('day_of_week')['streams'].sum().reset_index()
    dow_streams['day_of_week'] = pd.Categorical(dow_streams['day_of_week'], 
                                                  categories=dow_order, ordered=True)
    dow_streams = dow_streams.sort_values('day_of_week')
    
    fig = px.line(dow_streams, x='day_of_week', y='streams', markers=True)
    fig.update_traces(line_color='#ff6b9d', marker=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly trends
    st.subheader("📆 Monthly Streaming Trends")
    df['month'] = df['date'].dt.to_period('M').astype(str)
    monthly_streams = df.groupby('month')['streams'].sum().reset_index()
    
    fig = px.area(monthly_streams, x='month', y='streams',
                  labels={'streams': 'Total Streams', 'month': 'Month'})
    fig.update_traces(fillcolor='rgba(147, 197, 253, 0.5)', line_color='#3b82f6')
    st.plotly_chart(fig, use_container_width=True)

# ==================== DEMOGRAPHIC ANALYSIS ====================
elif page == "👥 Demographic Analysis":
    st.title("👥 Demographic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Streams by Age Group")
        age_streams = df.groupby('age_group')['streams'].sum().reset_index()
        age_order = ['13-17', '18-24', '25-34', '35-44', '45+']
        age_streams['age_group'] = pd.Categorical(age_streams['age_group'], 
                                                   categories=age_order, ordered=True)
        age_streams = age_streams.sort_values('age_group')
        
        fig = px.bar(age_streams, x='age_group', y='streams',
                     color='streams', color_continuous_scale='Mint')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Geographic Distribution")
        country_streams = df.groupby('country')['streams'].sum().reset_index()
        fig = px.pie(country_streams, values='streams', names='country',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    
    # Genre preferences by age
    st.subheader("🎼 Genre Preferences by Age Group")
    age_genre = df.groupby(['age_group', 'genre'])['streams'].sum().reset_index()
    age_order = ['13-17', '18-24', '25-34', '35-44', '45+']
    age_genre['age_group'] = pd.Categorical(age_genre['age_group'], 
                                            categories=age_order, ordered=True)
    age_genre = age_genre.sort_values('age_group')
    
    fig = px.bar(age_genre, x='age_group', y='streams', color='genre',
                 barmode='group', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)

# ==================== BEHAVIORAL METRICS ====================
elif page == "🎯 Behavioral Metrics":
    st.title("🎯 Behavioral Metrics")
    
    # Device usage
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📱 Device Distribution")
        device_streams = df.groupby('device')['streams'].sum().reset_index()
        fig = px.pie(device_streams, values='streams', names='device',
                     color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Average Listen Duration by Device")
        device_duration = df.groupby('device')['duration_sec'].mean().reset_index()
        device_duration['duration_min'] = device_duration['duration_sec'] / 60
        fig = px.bar(device_duration, x='device', y='duration_min',
                     color='duration_min', color_continuous_scale='Teal')
        st.plotly_chart(fig, use_container_width=True)
    
    # Listening patterns
    st.subheader("🔄 Listening Patterns")
    pattern_data = df.groupby(['time_of_day', 'device'])['streams'].sum().reset_index()
    
    fig = px.sunburst(pattern_data, path=['time_of_day', 'device'], values='streams',
                      color='streams', color_continuous_scale='RdYlBu_r')
    st.plotly_chart(fig, use_container_width=True)
    
    # Engagement metrics
    st.subheader("💫 Engagement Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_daily_streams = df.groupby('date')['streams'].sum().mean()
        st.metric("Avg Daily Streams", f"{avg_daily_streams:,.0f}")
    
    with col2:
        peak_hour = df.groupby('time_of_day')['streams'].sum().idxmax()
        st.metric("Peak Listening Time", peak_hour)
    
    with col3:
        most_used_device = df.groupby('device')['streams'].sum().idxmax()
        st.metric("Most Used Device", most_used_device)

# ==================== ML INTEGRATION ====================
elif page == "🤖 ML Integration":
    st.title("🤖 ML Integration")
    
    st.markdown("""
    ### 🔮 Machine Learning Features
    
    This section allows you to integrate ML APIs for advanced analytics:
    """)
    
    # API Configuration
    st.subheader("⚙️ API Configuration")
    
    api_provider = st.selectbox("Select ML Provider:", 
                                 ["OpenAI", "Anthropic Claude", "Google PaLM", "Hugging Face"])
    
    api_key = st.text_input("API Key:", type="password", 
                            help="Enter your API key for ML predictions")
    
    if st.button("Test Connection"):
        if api_key:
            st.success(f"✅ Connected to {api_provider} (Demo Mode)")
        else:
            st.error("Please enter an API key")
    
    st.markdown("---")
    
    # ML Features
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Prediction Models")
        st.markdown("""
        **Available Models:**
        - 🎵 Song Popularity Predictor
        - 👥 User Preference Analyzer
        - 📈 Trend Forecasting
        - 🎯 Recommendation Engine
        """)
        
        if st.button("Generate Predictions"):
            with st.spinner("Analyzing data..."):
                # Simulate predictions
                predictions = df.groupby('song')['streams'].sum().sort_values(ascending=False).head(5)
                st.success("✅ Predictions Generated!")
                
                st.markdown("**Top 5 Predicted Hits:**")
                for idx, (song, streams) in enumerate(predictions.items(), 1):
                    predicted_growth = np.random.uniform(5, 25)
                    st.markdown(f"{idx}. **{song}** - Predicted growth: +{predicted_growth:.1f}%")
    
    with col2:
        st.subheader("🧠 Insights Generation")
        st.markdown("""
        **AI-Powered Insights:**
        - 💡 Trend Analysis
        - 🎭 Sentiment Analysis
        - 📊 Market Segmentation
        - 🔍 Anomaly Detection
        """)
        
        if st.button("Generate Insights"):
            with st.spinner("Processing..."):
                st.success("✅ Insights Generated!")
                
                insights = [
                    "🎵 Pop genre shows 23% growth this month",
                    "👥 18-24 age group most engaged on mobile devices",
                    "⏰ Evening listening peaks at 7-9 PM",
                    "🌍 Strong growth in Asian markets (+35%)",
                    "🎤 Emerging artist 'New Wave' trending +150%"
                ]
                
                for insight in insights:
                    st.markdown(f"- {insight}")
    
    st.markdown("---")
    st.info("💡 **Note:** In production, integrate actual ML APIs for real-time predictions and insights.")

# ==================== UPLOAD CSV DATA ====================
elif page == "📤 Upload CSV Data":
    st.title("📤 Upload Your CSV Data")
    
    st.markdown("""
    ### 📋 Upload Instructions
    
    Your CSV file should contain the following columns:
    - `date` - Date of streaming (YYYY-MM-DD format)
    - `song` - Song name
    - `artist` - Artist name
    - `genre` - Music genre
    - `streams` - Number of streams
    - `duration_sec` - Song duration in seconds
    - `age_group` - Age group (e.g., 18-24, 25-34)
    - `country` - Country name
    - `device` - Device type (Mobile, Desktop, etc.)
    - `time_of_day` - Time period (Morning, Afternoon, Evening, Night)
    """)
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader("📁 Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the CSV
            new_data = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Successfully loaded **{len(new_data):,}** records!")
            
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(new_data):,}")
            with col2:
                st.metric("Total Columns", len(new_data.columns))
            with col3:
                file_size = uploaded_file.size / 1024  # KB
                st.metric("File Size", f"{file_size:.2f} KB")
            
            st.markdown("---")
            
            # Preview data
            st.subheader("📊 Data Preview")
            st.dataframe(new_data.head(20), use_container_width=True)
            
            # Data validation
            st.subheader("🔍 Data Validation")
            
            required_columns = ['date', 'song', 'artist', 'genre', 'streams']
            missing_cols = [col for col in required_columns if col not in new_data.columns]
            
            if missing_cols:
                st.warning(f"⚠️ Missing recommended columns: {', '.join(missing_cols)}")
            else:
                st.success("✅ All required columns present!")
            
            st.markdown("---")
            
            # Data statistics
            st.subheader("📈 Data Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Numeric Columns:**")
                numeric_cols = new_data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 0:
                    st.dataframe(new_data[numeric_cols].describe(), use_container_width=True)
                else:
                    st.info("No numeric columns found")
            
            with col2:
                st.markdown("**Categorical Columns:**")
                cat_cols = new_data.select_dtypes(include=['object']).columns.tolist()
                if len(cat_cols) > 0:
                    for col in cat_cols[:5]:  # Show first 5
                        unique_count = new_data[col].nunique()
                        st.markdown(f"- **{col}**: {unique_count} unique values")
                else:
                    st.info("No categorical columns found")
            
            st.markdown("---")
            
            # Action buttons
            st.subheader("💾 Save Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Replace Current Data", type="primary", use_container_width=True):
                    # Convert date column if exists
                    if 'date' in new_data.columns:
                        new_data['date'] = pd.to_datetime(new_data['date'], errors='coerce')
                    
                    st.session_state.streaming_data = new_data
                    st.success("✅ Data replaced successfully!")
                    st.balloons()
            
            with col2:
                if st.button("➕ Append to Current Data", use_container_width=True):
                    # Convert date column if exists
                    if 'date' in new_data.columns:
                        new_data['date'] = pd.to_datetime(new_data['date'], errors='coerce')
                    
                    st.session_state.streaming_data = pd.concat([
                        st.session_state.streaming_data, new_data
                    ], ignore_index=True)
                    st.success(f"✅ Added {len(new_data):,} records!")
            
            with col3:
                # Download processed data
                csv = new_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Processed",
                    data=csv,
                    file_name="processed_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.markdown("**Troubleshooting Tips:**")
            st.markdown("""
            - Ensure your file is a valid CSV
            - Check that dates are in YYYY-MM-DD format
            - Verify there are no special characters causing issues
            - Try opening the file in Excel/Google Sheets first
            """)
    
    else:
        # Show sample format
        st.info("👆 Upload a CSV file to get started")
        
        st.subheader("📋 Sample CSV Format")
        sample_data = {
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'song': ['Blinding Lights', 'Shape of You', 'Levitating'],
            'artist': ['The Weeknd', 'Ed Sheeran', 'Dua Lipa'],
            'genre': ['Pop', 'Pop', 'Pop'],
            'streams': [45000, 38000, 42000],
            'duration_sec': [200, 234, 203],
            'age_group': ['18-24', '25-34', '18-24'],
            'country': ['USA', 'UK', 'Canada'],
            'device': ['Mobile', 'Desktop', 'Mobile'],
            'time_of_day': ['Evening', 'Afternoon', 'Night']
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)
        
        # Download sample template
        csv_template = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Template",
            data=csv_template,
            file_name="streaming_data_template.csv",
            mime="text/csv"
        )

# ==================== DATA MANAGEMENT ====================
elif page == "📥 Data Management":
    st.title("📥 Data Management")
    
    tab1, tab2, tab3 = st.tabs(["💾 Current Data", "🔄 Generate Sample", "🗑️ Reset Data"])
    
    with tab1:
        st.subheader("Current Dataset")
        st.write(f"Total Records: {len(st.session_state.streaming_data):,}")
        st.dataframe(st.session_state.streaming_data.head(100), use_container_width=True)
        
        # Download button
        csv = st.session_state.streaming_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Current Data as CSV",
            data=csv,
            file_name="streaming_data.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("Generate Sample Data")
        num_records = st.slider("Number of records to generate:", 100, 10000, 1000)
        
        if st.button("Generate Data"):
            # Generate new sample data
            np.random.seed(int(datetime.now().timestamp()))
            dates = pd.date_range(start='2024-01-01', end='2024-11-13', freq='D')
            
            songs = ['Blinding Lights', 'Shape of You', 'Someone Like You', 'Levitating', 
                     'Watermelon Sugar', 'Bad Guy', 'Circles', 'Señorita', 'Sunflower', 'Rockstar']
            artists = ['The Weeknd', 'Ed Sheeran', 'Adele', 'Dua Lipa', 
                       'Harry Styles', 'Billie Eilish', 'Post Malone', 'Shawn Mendes', 'Post Malone', 'Post Malone']
            genres = ['Pop', 'Pop', 'Pop', 'Pop', 'Pop', 'Alternative', 'Hip-Hop', 'Pop', 'Hip-Hop', 'Hip-Hop']
            
            data = []
            for _ in range(num_records):
                idx = np.random.randint(0, len(songs))
                data.append({
                    'date': np.random.choice(dates),
                    'song': songs[idx],
                    'artist': artists[idx],
                    'genre': genres[idx],
                    'streams': np.random.randint(1000, 50000),
                    'duration_sec': np.random.randint(180, 300),
                    'age_group': np.random.choice(['13-17', '18-24', '25-34', '35-44', '45+']),
                    'country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'Germany', 'India']),
                    'device': np.random.choice(['Mobile', 'Desktop', 'Tablet', 'Smart Speaker']),
                    'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'])
                })
            
            st.session_state.streaming_data = pd.DataFrame(data)
            st.success(f"✅ Generated {num_records} new records!")
            st.rerun()
    
    with tab3:
        st.subheader("Reset to Default Data")
        st.warning("⚠️ This will delete all current data and restore the default sample dataset.")
        
        if st.button("Reset Data", type="primary"):
            # Reset to default sample data
            np.random.seed(42)
            dates = pd.date_range(start='2024-01-01', end='2024-11-13', freq='D')
            
            songs = ['Blinding Lights', 'Shape of You', 'Someone Like You', 'Levitating', 
                     'Watermelon Sugar', 'Bad Guy', 'Circles', 'Señorita', 'Sunflower', 'Rockstar']
            artists = ['The Weeknd', 'Ed Sheeran', 'Adele', 'Dua Lipa', 
                       'Harry Styles', 'Billie Eilish', 'Post Malone', 'Shawn Mendes', 'Post Malone', 'Post Malone']
            genres = ['Pop', 'Pop', 'Pop', 'Pop', 'Pop', 'Alternative', 'Hip-Hop', 'Pop', 'Hip-Hop', 'Hip-Hop']
            
            data = []
            for _ in range(5000):
                idx = np.random.randint(0, len(songs))
                data.append({
                    'date': np.random.choice(dates),
                    'song': songs[idx],
                    'artist': artists[idx],
                    'genre': genres[idx],
                    'streams': np.random.randint(1000, 50000),
                    'duration_sec': np.random.randint(180, 300),
                    'age_group': np.random.choice(['13-17', '18-24', '25-34', '35-44', '45+']),
                    'country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'Germany', 'India']),
                    'device': np.random.choice(['Mobile', 'Desktop', 'Tablet', 'Smart Speaker']),
                    'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'])
                })
            
            st.session_state.streaming_data = pd.DataFrame(data)
            st.success("✅ Data reset successfully!")
            st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Data Summary")
st.sidebar.metric("Total Records", f"{len(df):,}")
if len(date_range) == 2:
    st.sidebar.metric("Date Range", f"{(date_range[1] - date_range[0]).days} days")