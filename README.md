# Amazon-India---Sales-Analytics


# 🛍️ Amazon India Data Cleaning & Preparation (2015–2025)

## 📘 Project Overview
This project consolidates and cleans **Amazon India sales data** from **2015 to 2025** across multiple yearly CSV files.  
The final dataset — `Cleaned_Amazon_India_2015_2025.csv` — is fully standardized, analytics-ready, and suitable for both **business intelligence** and **machine learning** applications.

---

## 📂 Dataset Summary

- **Total Records:** 1,083,620  
- **Total Columns:** 34  
- **Period Covered:** 2015–2025  
- **File Size:** ~295 MB (CSV)  
- **Source Files:** `Amazon_India_2015.csv` … `Amazon_India_2025.csv`

---

## 🧹 Data Cleaning Pipeline

| Step | Description | Output |
|------|--------------|---------|
| **1. Merge Files** | Combined yearly CSVs (2015–2025) into one DataFrame. | Unified dataset |
| **2. Date Standardization** | Parsed mixed formats → ISO `YYYY-MM-DD`. | Consistent `datetime64` |
| **3. Price Cleaning** | Removed commas, text, and converted to numeric. | Float column `original_price_inr` |
| **4. Rating Normalization** | Parsed text ratings (“4.5 stars”) and filled missing. | Float 1.0–5.0 |
| **5. City Standardization** | Unified variants (“Bombay”→“Mumbai”). | Clean city names |
| **6. Boolean Columns** | Normalized `Yes/No`, `0/1`, `TRUE/FALSE`. | Consistent `True`/`False` |
| **7. Category Standardization** | Merged similar product categories. | Single canonical category |
| **8. Delivery Days Cleaning** | Parsed numeric text (“3-5 days”) → integers. | Int column `delivery_days` |
| **9. Duplicate Removal** | Removed duplicates on (`customer_id`, `product_name`, `order_date`). | 1,121,996 rows |
| **10. Outlier Handling** | Used IQR + decimal correction for unrealistic prices. | 1,083,620 rows |
| **11. Missing Value Handling** | Imputed missing key values. | Fully clean dataset |

---




## 📊 Usage Examples

### Import and Inspect
```python
import pandas as pd

df = pd.read_csv("Cleaned_Amazon_India_2015_2025.csv")
df.info()
df.head()
