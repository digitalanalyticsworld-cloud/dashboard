import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(page_title="Pulmonx Dashboard", layout="wide")

# --- Password Protection ---
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "test123test":  # Change this to your desired password
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.title("Pulmonx Dashboard Login")
        st.text_input(
            "Please enter the password to access the dashboard", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.title("Pulmonx Dashboard Login")
        st.text_input(
            "Please enter the password to access the dashboard", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

if not check_password():
    st.stop()  # Stop execution if password is wrong

# --- File Upload Section ---
st.sidebar.title("Data Files")
st.sidebar.info("Please upload the required data files to use this dashboard.")

paid_file = st.sidebar.file_uploader("Upload Paid Data CSV", type="csv")
organic_file = st.sidebar.file_uploader("Upload Organic Data CSV", type="csv")

# Check if files are uploaded
if not paid_file or not organic_file:
    st.warning("Please upload both data files in the sidebar to view the dashboard.")
    st.stop()  # Stop execution until files are uploaded

# --- Helper Functions for Data Cleaning ---
def clean_numeric_column(df, col_name, fill_value=0):
    """
    Cleans a numeric column by removing non-numeric characters and converting to float.
    Handles European-style numbers where '.' is thousands separator and ',' is decimal.
    """
    if col_name in df.columns:
        s = df[col_name].astype(str)
        # Remove currency symbol if present (e.g., '$')
        s = s.str.replace(r'[^\d\.,]+', '', regex=True) # Remove any non-digit, non-dot, non-comma characters
        
        # Heuristic to detect European format (comma as decimal separator)
        if s.str.contains(',').any() and not s.str.contains('\.').any():
             # If contains comma but no dot, assume comma is decimal separator
             s = s.str.replace(',', '.', regex=False)
        else:
            # Otherwise, assume dot is decimal and comma is thousands separator, or simply remove comma
            s = s.str.replace(',', '', regex=False) # Remove thousands separator
            
        # Convert to numeric, coercing errors to NaN, then fill NA
        df[col_name] = pd.to_numeric(s, errors='coerce').fillna(fill_value)
    return df

def clean_percentage_column(df, col_name, fill_value=0.0):
    """Cleans a percentage column by removing '%' and converting to float (0.0 to 1.0)."""
    if col_name in df.columns:
        s = df[col_name].astype(str)
        s = s.str.replace('%', '', regex=False)
        
        # Use pandas to_numeric for robustness, then divide by 100
        df[col_name] = pd.to_numeric(s, errors='coerce').fillna(fill_value) / 100
    return df

# --- Load and Clean Data ---
@st.cache_data # Cache data to avoid reloading on every rerun
def load_and_preprocess_data(paid_file, organic_file):
    # Load Paid Data
    try:
        paid_df = pd.read_csv(paid_file)
    except Exception as e:
        st.error(f"Error loading paid data: {e}")
        return None, None, None, None

    # Load Organic Data
    try:
        organic_df = pd.read_csv(organic_file)
    except Exception as e:
        st.error(f"Error loading organic data: {e}")
        return None, None, None, None

    # Clean Paid Data
    paid_metrics_to_clean = ['Impressions', 'Clicks', 'Reach', 'Total Spent', 'Average CPM', 'Average CPC', 'Total Engagements']
    for col in paid_metrics_to_clean:
        paid_df = clean_numeric_column(paid_df, col, fill_value=0)
    paid_df = clean_percentage_column(paid_df, 'Click Through Rate', fill_value=0.0) 

    # Clean Organic Data
    organic_metrics_to_clean = ['Impressions', 'Views', 'Offsite Views', 'Clicks', 'Likes', 'Comments', 'Reposts', 'Follows']
    for col in organic_metrics_to_clean:
        organic_df = clean_numeric_column(organic_df, col, fill_value=0)
    organic_df = clean_percentage_column(organic_df, 'Click through rate (CTR)', fill_value=0.0)
    organic_df = clean_percentage_column(organic_df, 'Engagement rate', fill_value=0.0)


    # Calculate Total Engagements for Organic data BEFORE aggregation
    organic_df['Calculated Total Engagements'] = (
        organic_df.get('Likes', 0) + 
        organic_df.get('Comments', 0) + 
        organic_df.get('Reposts', 0) + 
        organic_df.get('Clicks', 0) + 
        organic_df.get('Follows', 0)
    )

    # Aggregate Paid Data by Campaign Name
    paid_agg = paid_df.groupby('Campaign Name').agg(
        Impressions=('Impressions', 'sum'),
        Clicks=('Clicks', 'sum'),
        Reach=('Reach', 'sum'),
        Total_Spent=('Total Spent', 'sum'),
        Total_Engagements=('Total Engagements', 'sum'),
    ).reset_index()

    # Aggregate Organic Data by Post Title and Post Type
    organic_agg = organic_df.groupby(['Post title', 'Post type']).agg(
        Impressions=('Impressions', 'sum'),
        Views=('Views', 'sum'),
        Clicks=('Clicks', 'sum'),
        Likes=('Likes', 'sum'),
        Comments=('Comments', 'sum'),
        Reposts=('Reposts', 'sum'),
        Follows=('Follows', 'sum'),
        Total_Engagements=('Calculated Total Engagements', 'sum'),
    ).reset_index()
    
    return paid_agg, organic_agg, paid_df, organic_df # Return raw dataframes for content details

# Load the data from uploaded files
paid_agg_data, organic_agg_data, paid_raw_df, organic_raw_df = load_and_preprocess_data(paid_file, organic_file)

if paid_agg_data is None or organic_agg_data is None:
    st.stop()

# Initialize variables to hold the sum of filtered impressions and clicks
current_paid_impressions = 0
current_paid_clicks = 0
current_organic_impressions = 0
current_organic_clicks = 0

# --- Custom Components ---
def compact_multiselect(label, options, default, key=None, max_display=3):
    """
    Create a more compact multiselect that shows:
    - "All X" when all items are selected
    - Actual item names when only a few are selected
    - Count when many are selected
    """
    # Initialize session state for this specific multiselect
    if f"{key}_selections" not in st.session_state:
        st.session_state[f"{key}_selections"] = default
    
    # Create container for custom display
    container = st.container()
    
    # Get total count of options
    total_count = len(options)
    
    # For display purposes - determine what text to show
    current_selections = st.session_state[f"{key}_selections"]
    if len(current_selections) == total_count:
        display_text = f"All {label} ({total_count})"
    else:
        if len(current_selections) == 0:
            display_text = f"Select {label}..."
        elif len(current_selections) <= max_display:
            # Show the actual names when only a few are selected
            display_text = " | ".join(current_selections)
            if len(display_text) > 50:  # If text is too long, truncate
                display_text = display_text[:47] + "..."
        else:
            display_text = f"{len(current_selections)} {label} selected"
    
    # Create expander with the display text
    with container.expander(display_text, expanded=False):
        # Inside the expander, use a regular multiselect
        selected = st.multiselect(
            f"Select {label}",
            options=options,
            default=current_selections,
            key=key,
            label_visibility="collapsed"
        )
        
        # Store the selection in session state
        st.session_state[f"{key}_selections"] = selected
        
        # Add Select All / Clear buttons
        cols = st.columns([1, 1, 3])
        with cols[0]:
            if st.button("Select All", key=f"select_all_{key}"):
                st.session_state[f"{key}_selections"] = list(options)
                st.rerun()
        with cols[1]:
            if st.button("Clear", key=f"clear_{key}"):
                st.session_state[f"{key}_selections"] = []
                st.rerun()
    
    return st.session_state[f"{key}_selections"]

# --- Streamlit Dashboard Layout ---
# Apply custom CSS for background color, font, and metric card styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f9fafb;
        font-family: 'Segoe UI', 'Open Sans', sans-serif;
    }
    
    /* Main title */
    h1 {
        color: #4CAF50; /* Pulmonx green */
        font-size: 1.8em;
        font-weight: 600;
        margin-bottom: 0.5em;
        padding-bottom: 0.3em;
        border-bottom: 1px solid #eaecef;
    }
    
    /* Section headers */
    .section-header {
        color: white;
        padding: 8px 15px;
        border-radius: 4px;
        font-size: 1.1em;
        font-weight: 500;
        margin: 10px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .header-paid {
        background-color: #6366f1;
    }
    
    .header-organic {
        background-color: #0ea5e9;
    }
    
    .header-total {
        background-color: #4CAF50;
    }
    
    /* Metric cards */
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 15px;
    }
    
    .metric-card {
        background-color: white;
        border-radius: 4px;
        padding: 10px 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        flex: 1;
        min-width: 120px;
    }
    
    .metric-label {
        font-size: 0.8em;
        color: #6b7280;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-size: 1.5em;
        font-weight: 600;
        color: #111827;
    }
    
    /* Improved filter styling */
    .stExpander {
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        background-color: white;
    }
    
    .stExpander > div:first-child {
        padding: 8px 15px;
    }
    
    /* Style for the expander header */
    .stExpander > div:first-child > div:first-child > div:first-child > div:nth-child(2) > div:first-child {
        font-size: 0.9em;
        font-weight: normal;
    }
    
    /* Hide the full multiselect box initially */
    .stMultiSelect [data-baseweb="select"] [data-baseweb="tag"] {
        background-color: #e0e7ff;
        color: #4338ca;
        border-radius: 4px;
        margin: 2px;
    }
    
    /* Custom collapsible sections */
    .details-section {
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        margin-bottom: 15px;
        overflow: hidden;
    }
    
    .details-header {
        background-color: #f3f4f6;
        padding: 10px 15px;
        cursor: pointer;
        font-weight: 500;
        color: #374151;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .details-content {
        padding: 15px;
        background-color: white;
    }
    
    /* Override Streamlit defaults */
    div[data-testid="stVerticalBlock"] > div:has(div.stButton) {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    /* Remove extra padding around metrics */
    div[data-testid="stMetric"] {
        background-color: white;
        border-radius: 4px;
        padding: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.8em !important;
        color: #6b7280 !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.4em !important;
        font-weight: 600 !important;
    }
    
    /* Make multiselect more compact */
    div[data-baseweb="select"] {
        max-width: 100%;
    }
    
    /* Cleaner multiselect */
    div[data-baseweb="select"] > div {
        background-color: white;
        border-color: #d1d5db;
    }
    
    div[data-baseweb="select"] > div:hover {
        border-color: #6366f1;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #f3f4f6;
        color: #374151;
        border: 1px solid #d1d5db;
        padding: 2px 8px;
        font-size: 0.8em;
    }
    
    .stButton button:hover {
        background-color: #e5e7eb;
        border-color: #9ca3af;
    }
    
    /* Container for post details */
    .post-details-container {
        margin-top: 20px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
    }
    
    /* Filter headers in main display */
    .filter-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 15px;
        background-color: #6366f1;
        color: white;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    
    .filter-header.organic {
        background-color: #0ea5e9;
    }
    
    .filter-header-text {
        font-weight: 500;
    }
    
    .filter-info {
        font-size: 0.85em;
        opacity: 0.9;
    }
    
    /* Fix expander appearance */
    button[data-testid="baseButton-header"] {
        font-weight: normal !important;
        color: #374151 !important;
    }
    
    .stExpander {
        margin-bottom: 10px !important;
    }
    
    /* Style the sidebar file upload area */
    .stSidebar {
        background-color: #f8fafc;
    }
    
    .stSidebar [data-testid="stFileUploader"] {
        background-color: white;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }
    
    /* Make the login screen more centered and professional */
    div.stTitle {
        margin-top: 5rem;
        text-align: center;
    }
    
    div[data-testid="stForm"] {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main Dashboard Title
st.markdown("<h1>Pulmonx LinkedIn Dashboard</h1>", unsafe_allow_html=True)

# --- Main Dashboard Container ---
dashboard_container = st.container()

with dashboard_container:
    # --- Total Section at the top ---
    st.markdown("<div class='section-header header-total'>Total</div>", unsafe_allow_html=True)
    total_cols = st.columns(2)
    
    # Placeholder for total metrics (will be updated after filter processing)
    with total_cols[0]:
        total_impressions_metric = st.empty()
    with total_cols[1]:
        total_clicks_metric = st.empty()
    
    # --- Paid Section with Filter ---
    st.markdown("""
    <div class="filter-header">
        <div class="filter-header-text">Paid</div>
        <div class="filter-info">Filter by Campaign Name</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Compact dropdown filter for paid campaigns using custom function
    campaign_names_options = sorted(paid_agg_data['Campaign Name'].unique().tolist())
    selected_paid_campaigns = compact_multiselect(
        "Campaigns",
        options=campaign_names_options,
        default=campaign_names_options,
        key='paid_campaign_select'
    )
    
    # Process paid data based on filters
    filtered_paid_data = pd.DataFrame()
    if not selected_paid_campaigns:
        filtered_paid_data = paid_agg_data.iloc[0:0]  # Empty DataFrame with same structure
    else:
        filtered_paid_data = paid_agg_data[paid_agg_data['Campaign Name'].isin(selected_paid_campaigns)]
    
    # Display Paid Metrics
    if not filtered_paid_data.empty:
        current_paid_impressions = filtered_paid_data['Impressions'].sum()
        current_paid_clicks = filtered_paid_data['Clicks'].sum()
        total_paid_reach = filtered_paid_data['Reach'].sum()
        total_paid_engagements = filtered_paid_data['Total_Engagements'].sum()
        total_paid_spent = filtered_paid_data['Total_Spent'].sum()

        # Calculate derived metrics
        total_paid_ctr = (current_paid_clicks / current_paid_impressions) if current_paid_impressions > 0 else 0
        total_paid_cpc = (total_paid_spent / current_paid_clicks) if current_paid_clicks > 0 else 0
        total_paid_cpm = (total_paid_spent / current_paid_impressions * 1000) if current_paid_impressions > 0 else 0

        # Display metrics in a grid
        paid_row1 = st.columns(4)
        with paid_row1[0]: st.metric("Impressions", f"{current_paid_impressions:,.0f}")
        with paid_row1[1]: st.metric("Clicks", f"{current_paid_clicks:,.0f}")
        with paid_row1[2]: st.metric("Reach", f"{total_paid_reach:,.0f}")
        with paid_row1[3]: st.metric("CTR", f"{total_paid_ctr:.2%}")
        
        paid_row2 = st.columns(4)
        with paid_row2[0]: st.metric("Total Engagements", f"{total_paid_engagements:,.0f}")
        with paid_row2[1]: st.metric("CPC", f"${total_paid_cpc:,.2f}")
        with paid_row2[2]: st.metric("CPM", f"${total_paid_cpm:,.2f}")
        with paid_row2[3]: st.metric("Total Spent", f"${total_paid_spent:,.2f}")
    else:
        current_paid_impressions = 0
        current_paid_clicks = 0
        st.info("No data available for the selected paid campaign(s).")
    
    # --- Organic Section with Filters ---
    st.markdown("""
    <div class="filter-header organic">
        <div class="filter-header-text">Organic</div>
        <div class="filter-info">Filter by Post Title</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Pre-filter organic data to only show "Organic" post type
    organic_filtered_by_type = organic_agg_data[organic_agg_data['Post type'] == "Organic"]
    
    # Filter by post title using compact multiselect
    organic_titles_options = sorted(organic_filtered_by_type['Post title'].unique().tolist())
    selected_organic_posts = compact_multiselect(
        "Posts",
        options=organic_titles_options,
        default=organic_titles_options,
        key='organic_post_select'
    )
    
    # Process organic data based on filters
    filtered_organic_data = pd.DataFrame()
    if not selected_organic_posts:
        filtered_organic_data = organic_filtered_by_type.iloc[0:0]  # Empty DataFrame with same structure
    else:
        filtered_organic_data = organic_filtered_by_type[organic_filtered_by_type['Post title'].isin(selected_organic_posts)]
    
    # Display Organic Metrics
    if not filtered_organic_data.empty:
        current_organic_impressions = filtered_organic_data['Impressions'].sum()
        current_organic_clicks = filtered_organic_data['Clicks'].sum()
        total_organic_views = filtered_organic_data['Views'].sum()
        total_organic_reposts = filtered_organic_data['Reposts'].sum()
        total_organic_likes = filtered_organic_data['Likes'].sum()
        total_organic_comments = filtered_organic_data['Comments'].sum()
        
        # Calculate derived metrics
        total_organic_engagement_rate = (filtered_organic_data['Total_Engagements'].sum() / current_organic_impressions) if current_organic_impressions > 0 else 0
        total_organic_ctr = (current_organic_clicks / current_organic_impressions) if current_organic_impressions > 0 else 0

        # Display metrics in a grid
        organic_row1 = st.columns(4)
        with organic_row1[0]: st.metric("Impressions", f"{current_organic_impressions:,.0f}")
        with organic_row1[1]: st.metric("Clicks", f"{current_organic_clicks:,.0f}")
        with organic_row1[2]: st.metric("Engagement Rate", f"{total_organic_engagement_rate:.2%}")
        with organic_row1[3]: st.metric("CTR", f"{total_organic_ctr:.2%}")

        organic_row2 = st.columns(4)
        with organic_row2[0]: st.metric("Views", f"{total_organic_views:,.0f}")
        with organic_row2[1]: st.metric("Reposts", f"{total_organic_reposts:,.0f}")
        with organic_row2[2]: st.metric("Likes", f"{total_organic_likes:,.0f}")
        with organic_row2[3]: st.metric("Comments", f"{total_organic_comments:,.0f}")
    else:
        current_organic_impressions = 0
        current_organic_clicks = 0
        st.info("No data available for the selected Organic posts.")
    
    # Update total metrics now that we have calculated both paid and organic
    total_impressions = current_paid_impressions + current_organic_impressions
    total_clicks = current_paid_clicks + current_organic_clicks
    
    # Update the placeholders
    total_impressions_metric.metric("Total Impressions", f"{total_impressions:,.0f}")
    total_clicks_metric.metric("Total Clicks", f"{total_clicks:,.0f}")

# --- Post Details Section (below the dashboard) ---
st.markdown("<div class='post-details-container'>", unsafe_allow_html=True)
st.markdown("<h2>Content Details</h2>", unsafe_allow_html=True)

# Create tabs for Paid and Organic details
paid_tab, organic_tab = st.tabs(["Paid Campaign Details", "Organic Post Details"])

with paid_tab:
    if not selected_paid_campaigns:
        st.info("No paid campaign(s) selected to show details.")
    else:
        for campaign_name in selected_paid_campaigns:
            if campaign_name in paid_raw_df['Campaign Name'].unique().tolist():
                # Create an expander for each campaign
                with st.expander(f"Campaign: {campaign_name}"):
                    # Get campaign details
                    campaign_detail_row = paid_raw_df[paid_raw_df['Campaign Name'] == campaign_name].iloc[0]
                    
                    # Display campaign details in a clean format
                    details_cols = st.columns(2)
                    with details_cols[0]:
                        st.markdown(f"**Objective:** {campaign_detail_row.get('Campaign Objective', 'N/A')}")
                        st.markdown(f"**Type:** {campaign_detail_row.get('Campaign Type', 'N/A')}")
                        st.markdown(f"**Status:** {campaign_detail_row.get('Campaign Status', 'N/A')}")
                    
                    with details_cols[1]:
                        st.markdown(f"**Start Date:** {campaign_detail_row.get('Campaign Start Date', 'N/A')}")
                        st.markdown(f"**End Date:** {campaign_detail_row.get('Campaign End Date', 'N/A')}")
                    
                    # Display campaign dates
                    st.markdown("**Campaign Dates:**")
                    unique_dates = paid_raw_df[paid_raw_df['Campaign Name'] == campaign_name]['Date'].unique()
                    date_cols = st.columns(4)
                    for i, date_item in enumerate(sorted(unique_dates)):
                        date_cols[i % 4].markdown(f"• {date_item}")

with organic_tab:
    # Always filter to only show Organic post type
    selected_post_type = "Organic"
    
    if not selected_organic_posts:
        st.info(f"No {selected_post_type} post(s) selected to show details.")
    else:
        for post_title in selected_organic_posts:
            # Filter by both title and selected post type
            post_details_rows = organic_raw_df[
                (organic_raw_df['Post title'] == post_title) & 
                (organic_raw_df['Post type'] == selected_post_type)
            ]
            
            if not post_details_rows.empty:
                # Create an expander for each post
                with st.expander(f"Post: {post_title}"):
                    for index, row in post_details_rows.iterrows():
                        details_cols = st.columns(2)
                        with details_cols[0]:
                            st.markdown(f"**Post Type:** {row.get('Post type', 'N/A')}")
                            st.markdown(f"**Posted by:** {row.get('Posted by', 'N/A')}")
                            st.markdown(f"**Created Date:** {row.get('Created date', 'N/A')}")
                        
                        with details_cols[1]:
                            st.markdown(f"**Audience:** {row.get('Audience', 'N/A')}")
                            if pd.notna(row.get('Post link')):
                                st.markdown(f"**Link:** [View Post]({row['Post link']})")
                        
                        # Display post content if available
                        post_content = row.get('Post text', row.get('Post title', 'N/A'))
                        if pd.notna(post_content) and str(post_content) != 'N/A':
                            st.markdown("**Post Content:**")
                            st.markdown(f"> {str(post_content)[:300]}...")

st.markdown("</div>", unsafe_allow_html=True)

# Footer with update information
st.markdown(
    f"""
    <div style='margin-top: 20px; text-align: center; font-size: 0.8em; color: #6b7280;'>
        Last updated: {datetime.now().strftime("%B %d, %Y")} | Powered by Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
