import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Streamlit page
st.set_page_config(
    page_title="California House Price Map",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Cache data loading for better performance
@st.cache_data
def load_data():
    """Load and prepare the California housing dataset"""
    california_housing = fetch_california_housing(as_frame=True)
    df = california_housing.frame

    # Add price categories for color coding
    df['price_category'] = pd.cut(df['MedHouseVal'],
                                  bins=[0, 1.5, 3.0, 4.5, 6.0],
                                  labels=['Low ($0-150k)', 'Medium ($150-300k)',
                                          'High ($300-450k)', 'Very High ($450k+)'],
                                  include_lowest=True)

    # Create price per sqft approximation
    df['price_per_room'] = df['MedHouseVal'] / df['AveRooms']

    return df


@st.cache_data
def train_model(df):
    """Train the Random Forest model"""
    feature_columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                       'Population', 'AveOccup', 'Latitude', 'Longitude']

    X = df[feature_columns]
    y = df['MedHouseVal']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calculate metrics
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, r2, rmse, X_test, y_test, y_pred


def main():
    # Title and description
    st.title("California House Price Interactive Map")
    st.markdown(
        "Explore house prices across California with interactive visualizations and machine learning predictions.")

    # Load data
    with st.spinner("Loading California housing data..."):
        df = load_data()

    # Sidebar controls
    st.sidebar.header("Map Controls")

    # Sample size selector
    sample_size = st.sidebar.slider(
        "Sample Size (for performance)",
        min_value=1000,
        max_value=len(df),
        value=5000,
        step=1000,
        help="Reduce sample size for better performance"
    )

    # Price range filter
    min_price, max_price = st.sidebar.slider(
        "Price Range (in $100k)",
        min_value=float(df['MedHouseVal'].min()),
        max_value=float(df['MedHouseVal'].max()),
        value=(float(df['MedHouseVal'].min()), float(df['MedHouseVal'].max())),
        step=0.1
    )

    # Color scheme selector
    color_scheme = st.sidebar.selectbox(
        "Color Scheme",
        ["Viridis", "Plasma", "Inferno", "Turbo", "RdYlBu_r"],
        index=0
    )

    # Map type selector
    map_type = st.sidebar.selectbox(
        "Map Visualization",
        ["Scatter Plot", "Density Heatmap", "3D Surface"],
        index=0
    )

    # Filter data
    df_filtered = df[
        (df['MedHouseVal'] >= min_price) &
        (df['MedHouseVal'] <= max_price)
        ].sample(n=min(sample_size, len(df)), random_state=42)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("House Price Distribution Map")

        if map_type == "Scatter Plot":
            # Create scatter plot map
            fig = px.scatter_mapbox(
                df_filtered,
                lat="Latitude",
                lon="Longitude",
                color="MedHouseVal",
                size="Population",
                hover_data={
                    'MedInc': ':.2f',
                    'HouseAge': ':.1f',
                    'AveRooms': ':.1f',
                    'price_category': True,
                    'Population': ':,',
                    'MedHouseVal': ':.2f'
                },
                color_continuous_scale=color_scheme,
                size_max=15,
                zoom=5,
                height=600,
                title="California House Prices by Location"
            )

        elif map_type == "Density Heatmap":
            # Create density heatmap
            fig = px.density_mapbox(
                df_filtered,
                lat="Latitude",
                lon="Longitude",
                z="MedHouseVal",
                radius=10,
                zoom=5,
                color_continuous_scale=color_scheme,
                height=600,
                title="House Price Density Heatmap"
            )

        else:  # 3D Surface
            # Create 3D surface plot
            fig = go.Figure(data=[go.Scatter3d(
                x=df_filtered['Longitude'],
                y=df_filtered['Latitude'],
                z=df_filtered['MedHouseVal'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=df_filtered['MedHouseVal'],
                    colorscale=color_scheme,
                    opacity=0.8,
                    colorbar=dict(title="House Price ($100k)")
                ),
                text=df_filtered['price_category'],
                hovertemplate='<b>Price:</b> $%{z:.2f}00k<br>' +
                              '<b>Location:</b> (%{y:.2f}, %{x:.2f})<br>' +
                              '<b>Category:</b> %{text}<extra></extra>'
            )])

            fig.update_layout(
                title="3D House Price Distribution",
                scene=dict(
                    xaxis_title="Longitude",
                    yaxis_title="Latitude",
                    zaxis_title="Price ($100k)",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=600
            )

        if map_type != "3D Surface":
            fig.update_layout(
                mapbox_style="open-street-map",
                margin={"r": 0, "t": 50, "l": 0, "b": 0}
            )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Price Statistics")

        # Key metrics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "Avg Price",
                f"${df_filtered['MedHouseVal'].mean():.2f}00k",
                delta=f"{((df_filtered['MedHouseVal'].mean() / df['MedHouseVal'].mean() - 1) * 100):+.1f}%"
            )
            st.metric(
                "Max Price",
                f"${df_filtered['MedHouseVal'].max():.2f}00k"
            )

        with col_b:
            st.metric(
                "Min Price",
                f"${df_filtered['MedHouseVal'].min():.2f}00k"
            )
            st.metric(
                "Samples",
                f"{len(df_filtered):,}"
            )

        # Price distribution
        st.subheader("Price Distribution")
        price_dist = df_filtered['price_category'].value_counts()
        fig_pie = px.pie(
            values=price_dist.values,
            names=price_dist.index,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

        # Top expensive areas
        st.subheader("Most Expensive Areas")
        top_expensive = df_filtered.nlargest(5, 'MedHouseVal')[
            ['Latitude', 'Longitude', 'MedHouseVal', 'MedInc']
        ].round(2)

        for idx, row in top_expensive.iterrows():
            st.write(f"**${row['MedHouseVal']}00k** - ({row['Latitude']}, {row['Longitude']})")

    # Machine Learning Section
    st.header("Machine Learning Predictions")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Model Performance")
        with st.spinner("Training Random Forest model..."):
            model, r2, rmse, X_test, y_test, y_pred = train_model(df)

        # Display metrics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("R² Score", f"{r2:.4f}")
        with col_b:
            st.metric("RMSE", f"${rmse:.2f}00k")

        # Actual vs Predicted plot
        fig_scatter = px.scatter(
            x=y_test,
            y=y_pred,
            labels={'x': 'Actual Price ($100k)', 'y': 'Predicted Price ($100k)'},
            title="Actual vs Predicted Prices"
        )
        fig_scatter.add_shape(
            type="line", line=dict(dash="dash"),
            x0=y_test.min(), y0=y_test.min(),
            x1=y_test.max(), y1=y_test.max()
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.subheader("Let Us Make a Prediction")

        # Input features for prediction
        med_inc = st.number_input("Median Income", min_value=0.5, max_value=15.0, value=5.0, step=0.1)
        house_age = st.number_input("House Age", min_value=1.0, max_value=52.0, value=10.0, step=1.0)
        ave_rooms = st.number_input("Average Rooms", min_value=2.0, max_value=20.0, value=6.0, step=0.1)
        ave_bedrms = st.number_input("Average Bedrooms", min_value=0.5, max_value=5.0, value=1.2, step=0.1)
        population = st.number_input("Population", min_value=3, max_value=35000, value=3000, step=100)
        ave_occup = st.number_input("Average Occupancy", min_value=0.5, max_value=20.0, value=3.0, step=0.1)
        latitude = st.number_input("Latitude", min_value=32.5, max_value=42.0, value=34.0, step=0.1)
        longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-118.0, step=0.1)

        if st.button("Predict Price", type="primary"):
            # Make prediction
            features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms,
                                  population, ave_occup, latitude, longitude]])
            prediction = model.predict(features)[0]

            # Display prediction
            st.success(f"**Predicted Price: ${prediction:.2f}00k**")

            # Find similar properties
            df_similar = df[
                (abs(df['Latitude'] - latitude) < 1) &
                (abs(df['Longitude'] - longitude) < 1)
                ]['MedHouseVal']

            if len(df_similar) > 0:
                avg_nearby = df_similar.mean()
                st.info(f"Average nearby price: ${avg_nearby:.2f}00k")
                if prediction > avg_nearby:
                    st.write("Above average for the area")
                else:
                    st.write("Below average for the area")

    # Feature Importance
    st.subheader("Feature Importance")
    feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                     'Population', 'AveOccup', 'Latitude', 'Longitude']
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig_importance = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance in Random Forest Model"
    )
    st.plotly_chart(fig_importance, use_container_width=True)

    # Data Explorer
    with st.expander("Explore Raw Data"):
        st.dataframe(df_filtered.head(100))

        # Download button
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name="california_house_prices.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()