
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dateutil import parser

st.set_page_config(page_title="Analytics Dashboard", layout="wide")

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "Amazon_final_Dataset.csv"  # change if needed
DATE_COL = 'order_date'

# -----------------------------
# Utilities & Data Loading
# -----------------------------
@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # parse dates
    if DATE_COL in df.columns:
        try:
            df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        except Exception:
            df[DATE_COL] = df[DATE_COL].apply(lambda x: parser.parse(str(x), dayfirst=False) if pd.notna(x) else pd.NaT)

    # derive common time fields
    if DATE_COL in df.columns:
        df['order_year'] = df[DATE_COL].dt.year
        df['order_month'] = df[DATE_COL].dt.month
        df['order_quarter'] = df[DATE_COL].dt.to_period('Q').astype(str)

    # numeric conversions
    for col in ['original_price_inr','discount_percent','discounted_price_inr','quantity','subtotal_inr','delivery_charges','final_amount_inr','product_weight_kg','delivery_days']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # boolean fields
    for col in ['is_prime_member','is_festival_sale','is_prime_eligible']:
        if col in df.columns:
            # handle various representations
            df[col] = df[col].map(lambda x: True if str(x).strip().lower() in ['true','1','yes'] else (False if str(x).strip().lower() in ['false','0','no'] else np.nan))

    return df

@st.cache_data
def basic_aggregates(df):
    total_revenue = df['final_amount_inr'].sum() if 'final_amount_inr' in df.columns else 0
    customers = int(df['customer_id'].nunique()) if 'customer_id' in df.columns else 0
    orders = int(df['transaction_id'].nunique()) if 'transaction_id' in df.columns else len(df)
    aov = total_revenue / orders if orders>0 else 0
    if 'order_year' in df.columns and 'final_amount_inr' in df.columns:
        revenue_by_year = df.groupby('order_year')['final_amount_inr'].sum().sort_index()
        if len(revenue_by_year)>=2:
            last = revenue_by_year.iloc[-1]
            prev = revenue_by_year.iloc[-2]
            growth = (last - prev) / prev if prev!=0 else np.nan
        else:
            growth = np.nan
    else:
        revenue_by_year = pd.Series(dtype=float)
        growth = np.nan
    return {
        'total_revenue': total_revenue,
        'active_customers': customers,
        'orders': orders,
        'aov': aov,
        'yoy_growth': growth,
        'revenue_by_year': revenue_by_year
    }

# -----------------------------
# Dashboard Components (Pandas + Plotly only)
# -----------------------------

def executive_summary(df):
    st.title("Executive Summary")
    agg = basic_aggregates(df)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Revenue (INR)", f"{agg['total_revenue']:,.0f}")
    col2.metric("YoY Growth", f"{agg['yoy_growth']*100:.2f}%" if pd.notna(agg['yoy_growth']) else "-")
    col3.metric("Active Customers", f"{agg['active_customers']}")
    col4.metric("Average Order Value", f"{agg['aov']:,.2f}")

    # Top subcategories (by revenue)
    if 'subcategory' in df.columns and 'final_amount_inr' in df.columns:
        top_sub = df.groupby('subcategory')['final_amount_inr'].sum().sort_values(ascending=False).head(5)
        with col5:
            st.write("**Top Subcategories (by revenue)**")
            for c,v in top_sub.items():
                st.write(f"- {c}: ₹{v:,.0f}")
    else:
        col5.write("**Top Subcategories (by revenue)**")
        col5.write("No `subcategory` or `final_amount_inr` column found.")

    st.markdown("---")
    # Year-over-year chart
    if not agg['revenue_by_year'].empty:
        rby = agg['revenue_by_year'].reset_index()
        rby.columns = ['Year','Revenue']
        fig = px.bar(rby, x='Year', y='Revenue', title='Revenue by Year (YoY)')
        fig.add_trace(go.Scatter(x=rby['Year'], y=rby['Revenue'], mode='lines+markers', name='Trend'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Not enough yearly data to show YoY chart')

    # Monthly trend last 12 months
    if DATE_COL in df.columns and 'final_amount_inr' in df.columns:
        recent = df.set_index(DATE_COL).last('12M').resample('M')['final_amount_inr'].sum().reset_index()
        if not recent.empty:
            fig2 = px.line(recent, x=DATE_COL, y='final_amount_inr', title='Last 12 Months Revenue')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info('Not enough data for last 12 months trend')


def realtime_monitor(df):
    st.title("Real-time Business Performance Monitor")
    if DATE_COL not in df.columns:
        st.warning("No date information available.")
        return

    latest_date = df[DATE_COL].max()
    if pd.isna(latest_date):
        st.warning("No date information available.")
        return

    latest_period = latest_date.to_period('M')
    month_df = df[df[DATE_COL].dt.to_period('M')==latest_period]

    # Targets - simple UI
    st.sidebar.header("Targets (for current month)")
    rev_target = st.sidebar.number_input("Revenue target (INR)", value=int(month_df['final_amount_inr'].sum() * 1.1) if 'final_amount_inr' in month_df.columns else 0)
    cust_target = st.sidebar.number_input("New customers target", value=1000)

    rev = month_df['final_amount_inr'].sum() if 'final_amount_inr' in month_df.columns else 0
    new_customers = month_df['customer_id'].nunique() if 'customer_id' in month_df.columns else 0
    orders = month_df['transaction_id'].nunique() if 'transaction_id' in month_df.columns else len(month_df)

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Revenue ({latest_period.strftime('%b %Y')})", f"{rev:,.0f}", delta=f"{rev-rev_target:,.0f}")
    col2.metric("New Customers", f"{new_customers}", delta=f"{new_customers-cust_target}")
    col3.metric("Orders", f"{orders}")

    # Revenue run-rate (projected month-end based on day-of-month)
    today = latest_date
    day_of_month = today.day
    days_in_month = pd.Period(today, freq='M').days_in_month
    run_rate = (rev / day_of_month) * days_in_month if day_of_month>0 else rev
    st.write(f"**Projected month-end revenue (run-rate):** ₹{run_rate:,.0f}")

    # Alerts for underperformance
    alerts = []
    if rev_target>0 and rev < rev_target:
        alerts.append(f"Revenue below target by ₹{rev_target-rev:,.0f}")
    if cust_target>0 and new_customers < cust_target:
        alerts.append(f"New customers below target by {cust_target-new_customers}")
    if alerts:
        st.error(" | ".join(alerts))
    else:
        st.success("All monitored KPIs are on track for the current month")

    # Key operational indicators
    st.markdown("---")
    col4, col5 = st.columns(2)
    if 'delivery_days' in month_df.columns:
        avg_delivery = month_df['delivery_days'].mean()
        col4.metric("Average Delivery Days", f"{avg_delivery:.1f}")
    if 'customer_rating' in month_df.columns:
        avg_rating = month_df['customer_rating'].mean()
        col5.metric("Avg Customer Rating", f"{avg_rating:.2f}")


def strategic_overview(df):
    st.title("Strategic Overview")
    # Revenue share by subcategory
    if 'subcategory' in df.columns and 'final_amount_inr' in df.columns:
        sub_rev = df.groupby('subcategory')['final_amount_inr'].sum().reset_index().sort_values('final_amount_inr', ascending=False)
        fig = px.pie(sub_rev, names='subcategory', values='final_amount_inr', title='Revenue Share by Subcategory')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Need `subcategory` and `final_amount_inr` to show revenue share')

    # Geo expansion: top states
    if 'customer_state' in df.columns and 'final_amount_inr' in df.columns:
        state_rev = df.groupby('customer_state')['final_amount_inr'].sum().reset_index().sort_values('final_amount_inr', ascending=False).head(10)
        fig2 = px.bar(state_rev, x='customer_state', y='final_amount_inr', title='Top States by Revenue')
        st.plotly_chart(fig2, use_container_width=True)

    # Business health: quick ratios
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    orders = df['transaction_id'].nunique() if 'transaction_id' in df.columns else len(df)
    returns = df[df['return_status']==True].shape[0] if 'return_status' in df.columns else 0
    return_rate = returns / orders if orders>0 else np.nan
    col1.metric("Total Orders", orders)
    col2.metric("Return Rate", f"{return_rate*100:.2f}%" if pd.notna(return_rate) else "-")
    col3.metric("Active SKUs", df['product_id'].nunique() if 'product_id' in df.columns else 0)


def financial_performance(df):
    st.title("Financial Performance")
    # revenue breakdown by subcategory
    if 'subcategory' in df.columns and 'final_amount_inr' in df.columns:
        sub_rev = df.groupby('subcategory')['final_amount_inr'].sum().reset_index().sort_values('final_amount_inr', ascending=False)
        st.subheader("Revenue by Subcategory")
        st.plotly_chart(px.bar(sub_rev, x='subcategory', y='final_amount_inr'), use_container_width=True)
    else:
        st.info('Need `subcategory` and `final_amount_inr` for revenue breakdown')

    # profit margin analysis - approximate using assumed cost % of original price
    st.subheader("Profit Margin Analysis (approx)")
    assumed_cost_pct = st.slider("Assumed cost as % of original price", min_value=40, max_value=90, value=70)
    if 'original_price_inr' in df.columns and 'quantity' in df.columns and 'final_amount_inr' in df.columns:
        df = df.copy()
        df['assumed_cost'] = (df['original_price_inr'] * (assumed_cost_pct/100.0)) * df['quantity']
        df['profit'] = df['final_amount_inr'] - df['assumed_cost']
        margin_by_sub = df.groupby('subcategory').agg({'final_amount_inr':'sum','assumed_cost':'sum','profit':'sum'}).reset_index()
        margin_by_sub['margin_pct'] = margin_by_sub['profit'] / margin_by_sub['final_amount_inr']
        st.dataframe(margin_by_sub.sort_values('final_amount_inr', ascending=False).head(20))
    else:
        st.info('Need original_price_inr, quantity, and final_amount_inr for profit analysis')


def growth_analytics(df):
    st.title("Growth Analytics")
    # Customer growth
    if DATE_COL in df.columns and 'customer_id' in df.columns:
        cust_monthly = df.groupby(df[DATE_COL].dt.to_period('M'))['customer_id'].nunique().reset_index()
        cust_monthly[DATE_COL] = cust_monthly[DATE_COL].dt.to_timestamp()
        st.plotly_chart(px.line(cust_monthly, x=DATE_COL, y='customer_id', title='Unique Customers (monthly)'), use_container_width=True)

        # cumulative customers
        cum = cust_monthly.copy()
        cum['cumulative'] = cum['customer_id'].cumsum()
        st.plotly_chart(px.area(cum, x=DATE_COL, y='cumulative', title='Cumulative Customers'), use_container_width=True)
    else:
        st.info('Need order_date and customer_id for growth analytics')

    # Product portfolio expansion
    if DATE_COL in df.columns and 'product_id' in df.columns:
        sku_monthly = df.groupby(df[DATE_COL].dt.to_period('M'))['product_id'].nunique().reset_index()
        sku_monthly[DATE_COL] = sku_monthly[DATE_COL].dt.to_timestamp()
        st.plotly_chart(px.line(sku_monthly, x=DATE_COL, y='product_id', title='Active SKUs (monthly)'), use_container_width=True)
    else:
        st.info('Need order_date and product_id for SKU expansion')

# Revenue Analytics group

def revenue_trend(df):
    st.title("Revenue Trend Analysis")
    if DATE_COL not in df.columns or 'final_amount_inr' not in df.columns:
        st.info('Need order_date and final_amount_inr for revenue trend')
        return

    freq = st.selectbox("Granularity", ['M','Q','Y'])
    if freq=='M':
        series = df.set_index(DATE_COL).resample('M')['final_amount_inr'].sum().reset_index()
    elif freq=='Q':
        series = df.set_index(DATE_COL).resample('Q')['final_amount_inr'].sum().reset_index()
    else:
        series = df.set_index(DATE_COL).resample('Y')['final_amount_inr'].sum().reset_index()
    st.plotly_chart(px.line(series, x=DATE_COL, y='final_amount_inr', title='Revenue Trend'), use_container_width=True)

    if freq=='M':
        series['rolling12'] = series['final_amount_inr'].rolling(12, min_periods=1).mean()
        st.plotly_chart(px.line(series, x=DATE_COL, y=['final_amount_inr','rolling12'], title='Revenue + 12-month rolling'), use_container_width=True)


def category_performance(df):
    st.title("Subcategory Performance")
    if 'subcategory' not in df.columns or 'final_amount_inr' not in df.columns:
        st.info('Need subcategory and final_amount_inr for this analysis')
        return
    cat = df.groupby(['subcategory','order_year'])['final_amount_inr'].sum().reset_index()
    sel_cat = st.selectbox('Select subcategory', options=cat['subcategory'].unique())
    cdf = cat[cat['subcategory']==sel_cat]
    st.plotly_chart(px.bar(cdf, x='order_year', y='final_amount_inr', title=f'Revenue Trend - {sel_cat}'), use_container_width=True)
    st.dataframe(df[df['subcategory']==sel_cat].groupby('product_id').agg({'final_amount_inr':'sum','quantity':'sum'}).sort_values('final_amount_inr', ascending=False).head(50))


def geographic_revenue(df):
    st.title("Geographic Revenue Analysis")
    if 'customer_state' not in df.columns or 'final_amount_inr' not in df.columns:
        st.info('Need customer_state and final_amount_inr to show geographic revenue')
        return
    state = df.groupby('customer_state')['final_amount_inr'].sum().reset_index().sort_values('final_amount_inr', ascending=False)
    st.plotly_chart(px.bar(state.head(20), x='customer_state', y='final_amount_inr', title='Top States by Revenue'), use_container_width=True)


def festival_sales(df):
    st.title("Festival Sales Analytics")
    if 'is_festival_sale' not in df.columns or 'final_amount_inr' not in df.columns:
        st.info('Need is_festival_sale and final_amount_inr for festival analysis')
        return
    fest = df[df['is_festival_sale']==True]
    by_fest = fest.groupby('festival_name')['final_amount_inr'].sum().reset_index().sort_values('final_amount_inr', ascending=False)
    st.plotly_chart(px.bar(by_fest, x='festival_name', y='final_amount_inr', title='Revenue by Festival'), use_container_width=True)
   

def price_optimization(df):
    st.title("Price Optimization")
    st.markdown("This module gives simple price-elasticity-style insights using historical discount vs quantity patterns.")
    if 'discount_percent' in df.columns and 'quantity' in df.columns:
        pivot = df.groupby('discount_percent').agg({'quantity':'sum','final_amount_inr':'sum'}).reset_index()
        st.plotly_chart(px.scatter(pivot, x='discount_percent', y='quantity', size='final_amount_inr', title='Discount % vs Quantity Sold'), use_container_width=True)
        st.dataframe(pivot.sort_values('discount_percent').head(50))
    else:
        st.info('Need discount_percent and quantity columns for analysis')

# Customer Analytics group

def customer_segmentation(df):
    st.title("Customer Summary (RFM-like without clustering)")
    if 'customer_id' not in df.columns or DATE_COL not in df.columns or 'final_amount_inr' not in df.columns:
        st.info('Need customer_id, order_date and final_amount_inr for segmentation')
        return
    now = df[DATE_COL].max() + pd.Timedelta(days=1)
    r = df.groupby('customer_id')[DATE_COL].max().reset_index().rename(columns={DATE_COL:'last_purchase'})
    r['recency'] = (now - r['last_purchase']).dt.days
    f = df.groupby('customer_id').size().reset_index(name='frequency')
    m = df.groupby('customer_id')['final_amount_inr'].sum().reset_index(name='monetary')
    rfm = r.merge(f, on='customer_id').merge(m, on='customer_id')
    st.dataframe(rfm.sort_values('monetary', ascending=False).head(200))


def customer_journey(df):
    st.title("Customer Journey Analytics")
    if 'payment_method' in df.columns and 'customer_id' in df.columns:
        channel = df.groupby('payment_method').agg({'customer_id':'nunique','final_amount_inr':'sum'}).reset_index()
        st.plotly_chart(px.bar(channel, x='payment_method', y='customer_id', title='Customers by Payment Method'), use_container_width=True)
    else:
        st.info('Need payment_method and customer_id for this view')
   


def prime_membership(df):
    st.title('Prime Membership Analytics')
    if 'is_prime_member' not in df.columns or 'final_amount_inr' not in df.columns:
        st.info('Need is_prime_member and final_amount_inr')
        return
    prime = df.groupby('is_prime_member').agg({'customer_id':'nunique','final_amount_inr':'sum'}).reset_index()
    st.plotly_chart(px.pie(prime, names='is_prime_member', values='final_amount_inr', title='Revenue: Prime vs Non-Prime'), use_container_width=True)


def customer_retention(df):
    st.title('Customer Retention & Cohorts')
    if 'customer_id' not in df.columns or DATE_COL not in df.columns:
        st.info('Need customer_id and order_date for cohorts')
        return
    df = df.copy()
    df['cohort_month'] = df.groupby('customer_id')[DATE_COL].transform('min').dt.to_period('M').dt.to_timestamp()
    cohort = df.groupby(['cohort_month', df[DATE_COL].dt.to_period('M').dt.to_timestamp()])['customer_id'].nunique().reset_index()
    cohort_pivot = cohort.pivot(index='cohort_month', columns=DATE_COL, values='customer_id').fillna(0)
    st.dataframe(cohort_pivot.head(20))
   


def demographics_behavior(df):
    st.title('Demographics & Behavior')
    if 'customer_age_group' in df.columns and 'final_amount_inr' in df.columns:
        age = df.groupby('customer_age_group')['final_amount_inr'].sum().reset_index()
        st.plotly_chart(px.bar(age, x='customer_age_group', y='final_amount_inr', title='Revenue by Age Group'), use_container_width=True)
    if 'customer_tier' in df.columns and 'final_amount_inr' in df.columns:
        tier = df.groupby('customer_tier')['final_amount_inr'].sum().reset_index()
        st.plotly_chart(px.pie(tier, names='customer_tier', values='final_amount_inr', title='Revenue by Customer Tier'), use_container_width=True)

# Product & Inventory

def product_performance(df):
    st.title('Product Performance')
    cols = ['product_id','product_name','final_amount_inr','quantity','customer_rating']
    present = [c for c in cols if c in df.columns]
    if not present:
        st.info('Need product-level columns (product_id/product_name/final_amount_inr)')
        return
    prod = df.groupby([c for c in ['product_id','product_name'] if c in df.columns]).agg({k:'sum' for k in ['final_amount_inr','quantity'] if k in df.columns}).reset_index()
    prod = prod.sort_values('final_amount_inr', ascending=False) if 'final_amount_inr' in prod.columns else prod
    st.dataframe(prod.head(200))


def brand_analytics(df):
    st.title('Brand Analytics')
    if 'brand' not in df.columns or 'final_amount_inr' not in df.columns:
        st.info('Need brand and final_amount_inr')
        return
    brand = df.groupby('brand')['final_amount_inr'].sum().reset_index().sort_values('final_amount_inr', ascending=False)
    st.plotly_chart(px.bar(brand.head(30), x='brand', y='final_amount_inr', title='Top Brands by Revenue'), use_container_width=True)


def inventory_optimization(df):
    st.title('Inventory Optimization')
    if 'product_id' not in df.columns:
        st.info('Need product_id')
        return
    sku = df.groupby(['product_id']).agg({'quantity':'sum','final_amount_inr':'sum'}).reset_index().sort_values('quantity', ascending=False)
    st.dataframe(sku.head(200))
  


def ratings_reviews(df):
    st.title('Ratings & Reviews')
    if 'customer_rating' not in df.columns:
        st.info('No customer_rating column')
        return
    rating_dist = df['customer_rating'].value_counts().reset_index().sort_values('index')
    st.plotly_chart(px.bar(rating_dist, x='index', y='customer_rating', labels={'index':'Rating','customer_rating':'Count'}, title='Rating Distribution'), use_container_width=True)


def new_product_launch(df):
    st.title('New Product Launch')
    st.markdown('Track first 30/60/90 days performance for new SKUs')
    if 'product_id' not in df.columns or DATE_COL not in df.columns:
        st.info('Need product_id and order_date')
        return
    first_sale = df.groupby('product_id')[DATE_COL].min().reset_index()
    first_sale.columns = ['product_id','first_sale_date']
    merged = df.merge(first_sale, on='product_id', how='left')
    merged['days_since_launch'] = (merged[DATE_COL] - merged['first_sale_date']).dt.days
    launch_perf = merged[merged['days_since_launch']<=90].groupby('product_id').agg({'final_amount_inr':'sum','quantity':'sum'}).reset_index().sort_values('final_amount_inr', ascending=False)
    st.dataframe(launch_perf.head(200))

# Operations & Logistics

def delivery_performance(df):
    st.title('Delivery Performance')
    if 'delivery_days' not in df.columns or 'customer_state' not in df.columns:
        st.info('Need delivery_days and customer_state')
        return
    dp = df.groupby('customer_state')['delivery_days'].median().reset_index().sort_values('delivery_days')
    st.plotly_chart(px.bar(dp.head(30), x='customer_state', y='delivery_days', title='Median Delivery Days by State'), use_container_width=True)


def payment_analytics(df):
    st.title('Payment Analytics')
    if 'payment_method' in df.columns:
        pm = df.groupby('payment_method').agg({'transaction_id':'nunique' if 'transaction_id' in df.columns else 'size','final_amount_inr':'sum' if 'final_amount_inr' in df.columns else 'size'}).reset_index()
        st.plotly_chart(px.bar(pm, x='payment_method', y='final_amount_inr', title='Revenue by Payment Method'), use_container_width=True)
    else:
        st.info('Need payment_method')


def customer_service(df):
    st.title('Customer Service')
    
    if 'customer_rating' in df.columns:
        st.metric('Avg Rating', f"{df['customer_rating'].mean():.2f}")

# Note: The following functions were removed as requested:
# - market_intelligence
# - cross_selling
# - supply_chain
# - returns_cancellations

# Advanced Analytics (Descriptive only)

def predictive_analytics(df):
    st.title('Predictive Analytics (Descriptive Proxies)')
   
    if 'customer_id' not in df.columns or DATE_COL not in df.columns:
        st.info('Need customer_id and order_date for churn proxies')
        return
    now = df[DATE_COL].max()
    last_order = df.groupby('customer_id')[DATE_COL].max().reset_index()
    last_order['days_since'] = (now - last_order[DATE_COL]).dt.days
    churn_proxy = (last_order['days_since']>90).mean()
    st.metric('Estimated churn proxy (customers with >90 days since last order)', f"{churn_proxy*100:.2f}%")


def seasonal_planning(df):
    st.title('Seasonal Planning')
    st.markdown('Show monthly seasonality and festival peaks')
    if DATE_COL in df.columns and 'final_amount_inr' in df.columns:
        m = df.set_index(DATE_COL).resample('M')['final_amount_inr'].sum().reset_index()
        m['month'] = m[DATE_COL].dt.month
        season = m.groupby('month')['final_amount_inr'].mean().reset_index()
        st.plotly_chart(px.bar(season, x='month', y='final_amount_inr', title='Avg Revenue by Month'), use_container_width=True)
    else:
        st.info('Need order_date and final_amount_inr')


def command_center(df):
    st.title('Business Intelligence Command Center')
    st.markdown('Integrates key KPIs and simple rule-based alerts')
    agg = basic_aggregates(df)
    st.metric('Total Revenue', f"{agg['total_revenue']:,.0f}")
    st.metric('Active Customers', f"{agg['active_customers']}")
    if pd.notna(agg['yoy_growth']) and agg['yoy_growth']<0:
        st.error('Revenue YoY is negative — investigate')
    else:
        st.success('YoY growth positive or insufficient data')

# -----------------------------
# Main App Layout
# -----------------------------

def main():
    st.sidebar.title('Amazon Sales Analytics Dashboard')
    section = st.sidebar.selectbox('Choose dashboard group', [
        'Executive', 'Revenue ', 'Customer', 'Product & Inventory',
        'Operations', 'Advanced'
    ])

    df = load_data()

    if section=='Executive':
        page = st.sidebar.selectbox('Select', ['Executive Summary ', 'Real-time Monitor', 'Strategic Overview', 'Financial Performance', 'Growth Analytics'])
        if page.startswith('Executive Summary'):
            executive_summary(df)
        elif page.startswith('Real-time Monitor'):
            realtime_monitor(df)
        elif page.startswith('Strategic Overview'):
            strategic_overview(df)
        elif page.startswith('Financial Performance'):
            financial_performance(df)
        elif page.startswith('Growth Analytics'):
            growth_analytics(df)

    elif section=='Revenue ':
        page = st.sidebar.selectbox('Select', ['Revenue Trend', 'Subcategory Performance', 'Geographic Revenue', 'Festival Sales', 'Price Optimization'])
        if page.startswith('Revenue Trend'):
            revenue_trend(df)
        elif page.startswith('Subcategory Performance'):
            category_performance(df)
        elif page.startswith('Geographic'):
            geographic_revenue(df)
        elif page.startswith('Festival'):
            festival_sales(df)
        elif page.startswith('Price'):
            price_optimization(df)

    elif section=='Customer ':
        page = st.sidebar.selectbox('Select', ['Customer Summary ', 'Customer Journey ', 'Prime Analytics ', 'Retention ', 'Demographics'])
        if page.startswith('Customer Summary'):
            customer_segmentation(df)
        elif page.startswith('Customer Journey'):
            customer_journey(df)
        elif page.startswith('Prime'):
            prime_membership(df)
        elif page.startswith('Retention'):
            customer_retention(df)
        elif page.startswith('Demographics'):
            demographics_behavior(df)

    elif section=='Product & Inventory':
        page = st.sidebar.selectbox('Select', ['Product Performance ', 'Brand Analytics ', 'Inventory Optimization ', 'Ratings & Reviews ', 'New Product Launch '])
        if page.startswith('Product Performance'):
            product_performance(df)
        elif page.startswith('Brand Analytics'):
            brand_analytics(df)
        elif page.startswith('Inventory'):
            inventory_optimization(df)
        elif page.startswith('Ratings'):
            ratings_reviews(df)
        elif page.startswith('New Product'):
            new_product_launch(df)

    elif section=='Operations':
        page = st.sidebar.selectbox('Select', ['Delivery Performance', 'Payment Analytics', 'Customer Service '])
        if page.startswith('Delivery'):
            delivery_performance(df)
        elif page.startswith('Payment'):
            payment_analytics(df)
        elif page.startswith('Customer Service'):
            customer_service(df)

    elif section=='Advanced':
        page = st.sidebar.selectbox('Select', ['Descriptive Proxies', 'Seasonal Planning', 'Command Center'])
        if page.startswith('Descriptive'):
            predictive_analytics(df)
        elif page.startswith('Seasonal'):
            seasonal_planning(df)
        elif page.startswith('Command'):
            command_center(df)


if __name__=='__main__':
    main()
