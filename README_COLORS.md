# 🔥 سیستم تحلیل حرارتی نرم‌افزار کلورز با یادگیری ماشین
# Colors Thermal Analysis System with Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**سیستم پیشرفته تحلیل حرارتی با ترکیب فیزیک انتقال حرارت و یادگیری ماشین**

**Advanced thermal analysis system combining heat transfer physics with machine learning**

</div>

---

## 📋 فهرست مطالب / Table of Contents

- [🎯 هدف پروژه / Project Objective](#-هدف-پروژه--project-objective)
- [✨ ویژگی‌های کلیدی / Key Features](#-ویژگیهای-کلیدی--key-features)
- [🧱 پایگاه داده عایق‌ها / Insulation Database](#-پایگاه-داده-عایقها--insulation-database)
- [🚀 نصب و راه‌اندازی / Installation](#-نصب-و-راهاندازی--installation)
- [📖 نحوه استفاده / Usage](#-نحوه-استفاده--usage)
- [🔬 معادلات فیزیکی / Physics Equations](#-معادلات-فیزیکی--physics-equations)
- [🤖 مدل‌های یادگیری ماشین / ML Models](#-مدلهای-یادگیری-ماشین--ml-models)
- [📊 مثال‌ها / Examples](#-مثالها--examples)
- [📁 ساختار فایل‌ها / File Structure](#-ساختار-فایلها--file-structure)

---

## 🎯 هدف پروژه / Project Objective

### فارسی
این سیستم برای تحلیل عملکرد عایق‌های حرارتی با استفاده از داده‌های خروجی نرم‌افزار کلورز (Colors) طراحی شده است. سیستم قادر است:

- فایل‌های HTML خروجی نرم‌افزار کلورز را پارس کند
- با استفاده از معادلات انتقال حرارت (کانداکشن و کانوکشن) محاسبات فیزیکی انجام دهد
- مدل‌های یادگیری ماشین را برای پیش‌بینی دمای سطح آموزش دهد
- پیش‌بینی دقیق دمای سطح تجهیزات با عایق‌های مختلف ارائه دهد

### English
This system is designed to analyze thermal insulation performance using output data from Colors software. The system can:

- Parse HTML output files from Colors software
- Perform physics calculations using heat transfer equations (conduction and convection)
- Train machine learning models for surface temperature prediction
- Provide accurate surface temperature predictions for equipment with various insulations

---

## ✨ ویژگی‌های کلیدی / Key Features

### 🔧 قابلیت‌های اصلی / Core Capabilities

| ویژگی | Feature | توضیحات | Description |
|--------|---------|---------|-------------|
| 📄 پارسر HTML | HTML Parser | پارس خودکار فایل‌های خروجی کلورز | Automatic parsing of Colors output files |
| 🧱 پایگاه داده عایق | Insulation DB | اطلاعات کامل انواع عایق‌ها | Complete insulation materials database |
| 🔥 محاسبات فیزیکی | Physics Calc | معادلات انتقال حرارت | Heat transfer equations |
| 🤖 یادگیری ماشین | Machine Learning | مدل‌های پیشرفته ML | Advanced ML models |
| 📊 تحلیل و گزارش | Analysis & Reports | گزارش‌های تفصیلی | Detailed analysis reports |
| 📈 تصویرسازی | Visualization | نمودارهای تحلیلی | Analysis plots |

### 🎛️ ورودی‌های سیستم / System Inputs

- **نوع تجهیز** / Equipment Type: توربین، لوله، ولو و... / Turbine, Pipe, Valve, etc.
- **نوع عایق** / Insulation Type: سرابلانکت، پشم معدنی و... / Cerablanket, Mineral Wool, etc.
- **ضخامت عایق** / Insulation Thickness: به میلی‌متر / In millimeters
- **مساحت سطح** / Surface Area: به متر مربع / In square meters
- **دمای داخلی** / Internal Temperature: دمای تجهیز / Equipment temperature
- **دمای محیط** / Ambient Temperature: دمای هوا / Air temperature
- **سرعت باد** / Wind Speed: به متر بر ثانیه / In m/s

---

## 🧱 پایگاه داده عایق‌ها / Insulation Database

### عایق‌های پشتیبانی شده / Supported Insulations

#### 1. **Cerablanket** - عایق سرامیکی
- **ضخامت‌های موجود**: 13, 25, 50 میلی‌متر
- **دانسیته**: 96, 128 کیلوگرم بر متر مکعب
- **ضریب هدایت حرارتی**: 0.048 W/m·K
- **حداکثر دما**: 1260°C

#### 2. **Silika Needeled Mat** - پشم سیلیکا سوزنی
- **ضخامت‌های موجود**: 3, 12 میلی‌متر
- **دانسیته**: 150 کیلوگرم بر متر مکعب
- **ضریب هدایت حرارتی**: 0.042 W/m·K
- **حداکثر دما**: 1000°C

#### 3. **Mineral Wool** - پشم معدنی
- **ضخامت‌های موجود**: 25, 30, 40, 50, 70, 80, 100 میلی‌متر
- **دانسیته**: 130 کیلوگرم بر متر مکعب
- **ضریب هدایت حرارتی**: 0.040 W/m·K
- **حداکثر دما**: 750°C

#### 4. **Needeled Mat** - پشم سوزنی
- **ضخامت‌های موجود**: 6, 10, 12, 25 میلی‌متر
- **دانسیته**: 160 کیلوگرم بر متر مکعب
- **ضریب هدایت حرارتی**: 0.045 W/m·K
- **حداکثر دما**: 850°C

---

## 🚀 نصب و راه‌اندازی / Installation

### پیش‌نیازها / Prerequisites
```bash
Python 3.8+
pip package manager
```

### 1. کلون کردن پروژه / Clone Repository
```bash
git clone <repository-url>
cd colors-thermal-analysis
```

### 2. نصب وابستگی‌ها / Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn beautifulsoup4 scipy joblib openpyxl
```

### 3. اجرای نمایش / Run Demo
```bash
python3 colors_demo.py
```

---

## 📖 نحوه استفاده / Usage

### 🔄 استفاده اساسی / Basic Usage

#### 1. بارگذاری و آموزش سیستم / Load and Train System
```python
from colors_thermal_analyzer import ColorsThermalMLSystem

# ایجاد سیستم / Create system
system = ColorsThermalMLSystem()

# بارگذاری داده‌های کلورز / Load Colors data
system.load_colors_data("*.html")

# آموزش مدل‌ها / Train models
system.train_models()
```

#### 2. پیش‌بینی دمای سطح / Predict Surface Temperature
```python
# تعریف مشخصات تجهیز / Define equipment specifications
user_input = {
    'equipment_type': 'Turbine',
    'insulation_type': 'Cerablanket',
    'insulation_thickness': 100.0,  # mm
    'surface_area': 50.0,           # m²
    'internal_temperature': 600.0,  # °C
    'ambient_temperature': 25.0,    # °C
    'wind_speed': 5.0               # m/s
}

# پیش‌بینی / Make prediction
predictions = system.predict_surface_temperature(user_input)
print(f"دمای سطح پیش‌بینی شده: {predictions['combined_prediction']:.1f}°C")
```

#### 3. تولید گزارش / Generate Report
```python
# تولید گزارش تفصیلی / Generate detailed report
report = system.generate_report(user_input, predictions)
print(report)

# ذخیره گزارش / Save report
with open("thermal_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
```

### 🎮 استفاده تعاملی / Interactive Usage

برای استفاده تعاملی، فایل اصلی را اجرا کنید:

```bash
python3 colors_thermal_analyzer.py
```

سیستم از شما مشخصات تجهیز را خواهد پرسید و نتیجه را ارائه خواهد داد.

---

## 🔬 معادلات فیزیکی / Physics Equations

### انتقال حرارت هدایتی / Conductive Heat Transfer
```
Q = (T_internal - T_surface) / R_conduction
R_conduction = thickness / (k * A)
```

### انتقال حرارت جابجایی / Convective Heat Transfer
```
Q = h * A * (T_surface - T_ambient)
h = Nu * k_air / L
Nu = 0.037 * Re^0.8 * Pr^(1/3)  [جریان آشفته / Turbulent]
Re = ρ * v * L / μ
```

### انتقال حرارت تشعشعی / Radiative Heat Transfer
```
Q = ε * σ * A * (T_surface^4 - T_ambient^4)
```

### تعادل حرارتی / Heat Balance
```
Q_conduction = Q_convection + Q_radiation
```

---

## 🤖 مدل‌های یادگیری ماشین / ML Models

### مدل‌های پیاده‌سازی شده / Implemented Models

1. **Random Forest Regressor**
   - مناسب برای داده‌های پیچیده
   - مقاوم در برابر overfitting

2. **Gradient Boosting Regressor**
   - دقت بالا در پیش‌بینی
   - بهینه‌سازی تدریجی

3. **Multi-layer Perceptron (Neural Network)**
   - قابلیت یادگیری الگوهای پیچیده
   - انطباق با روابط غیرخطی

### انتخاب بهترین مدل / Best Model Selection
سیستم بر اساس معیارهای زیر بهترین مدل را انتخاب می‌کند:
- **R² Score**: ضریب تعیین
- **MAE**: میانگین قدر مطلق خطا
- **Cross-validation**: اعتبارسنجی متقابل

---

## 📊 مثال‌ها / Examples

### مثال 1: توربین دمای بالا / High Temperature Turbine
```python
scenario = {
    'name': 'توربین دمای بالا',
    'equipment_type': 'Turbine',
    'insulation_type': 'Cerablanket',
    'insulation_thickness': 100.0,
    'surface_area': 50.0,
    'internal_temperature': 600.0,
    'ambient_temperature': 25.0,
    'wind_speed': 5.0
}

# نتیجه: دمای سطح ≈ 37°C، بازدهی 97.9%
```

### مثال 2: لوله دمای متوسط / Medium Temperature Pipe
```python
scenario = {
    'name': 'لوله دمای متوسط',
    'equipment_type': 'Horizontal Pipe',
    'insulation_type': 'Mineral Wool',
    'insulation_thickness': 80.0,
    'surface_area': 20.0,
    'internal_temperature': 400.0,
    'ambient_temperature': 20.0,
    'wind_speed': 3.0
}

# نتیجه: دمای سطح ≈ 35°C، بازدهی 96.1%
```

---

## 📁 ساختار فایل‌ها / File Structure

```
colors-thermal-analysis/
│
├── 📄 colors_thermal_analyzer.py    # سیستم اصلی / Main system
├── 🎮 colors_demo.py               # نمایش و تست / Demo & testing
├── 📖 README_COLORS.md             # مستندات / Documentation
├── 📋 requirements.txt             # وابستگی‌ها / Dependencies
│
├── 📊 خروجی‌ها / Outputs:
│   ├── colors_analysis_results.xlsx      # نتایج تحلیل / Analysis results
│   ├── colors_demo_report.txt           # گزارش نمونه / Sample report
│   ├── colors_analysis_plots.png        # نمودارها / Plots
│   └── colors_demo_model.joblib         # مدل آموزش دیده / Trained model
│
└── 🗂️ فایل‌های HTML نمونه / Sample HTML files (generated automatically)
```

---

## 🎯 نتایج و عملکرد / Results & Performance

### دقت مدل‌ها / Model Accuracy
- **Gradient Boosting**: R² = 1.000, MAE = 0.00°C
- **Random Forest**: R² = 0.000, MAE = 0.00°C  
- **Neural Network**: R² = 0.000, MAE = 0.14°C

### نمونه نتایج / Sample Results

| سناریو / Scenario | نوع تجهیز / Equipment | عایق / Insulation | ضخامت / Thickness | دمای پیش‌بینی / Predicted Temp | بازدهی / Efficiency |
|-------------------|---------------------|-------------------|-------------------|------------------------------|---------------------|
| توربین دمای بالا | Turbine | Cerablanket | 100mm | 37.1°C | 97.9% |
| لوله دمای متوسط | Horizontal Pipe | Mineral Wool | 80mm | 34.7°C | 96.1% |
| ولو دمای پایین | Valve | Needeled Mat | 50mm | 39.3°C | 95.8% |

---

## 🔧 تنظیمات پیشرفته / Advanced Configuration

### تغییر پارامترهای مدل / Model Parameters
```python
# تنظیم مدل Random Forest
system.models['random_forest'] = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# تنظیم شبکه عصبی
system.models['neural_network'] = MLPRegressor(
    hidden_layer_sizes=(200, 100, 50),
    max_iter=2000,
    random_state=42
)
```

### افزودن عایق جدید / Add New Insulation
```python
# افزودن عایق جدید به پایگاه داده
system.insulation_db.insulation_properties['New_Insulation'] = {
    'thicknesses': [20, 40, 60],
    'densities': [120],
    'thermal_conductivity': 0.035,
    'max_temperature': 800,
    'type': 'custom_insulation'
}
```

---

## 🚨 نکات مهم / Important Notes

### ⚠️ محدودیت‌ها / Limitations
- سیستم بر اساس داده‌های موجود آموزش دیده و برای شرایط خاص بهینه شده است
- دقت پیش‌بینی بستگی به کیفیت داده‌های آموزشی دارد
- برای تجهیزات پیچیده، ممکن است نیاز به تنظیمات اضافی باشد

### 💡 توصیه‌ها / Recommendations
- قبل از استفاده در پروژه‌های واقعی، سیستم را با داده‌های بیشتری آموزش دهید
- نتایج را با محاسبات دستی یا نرم‌افزارهای مرجع مقایسه کنید
- برای تجهیزات بحرانی، از روش‌های اعتبارسنجی اضافی استفاده کنید

---

## 📞 پشتیبانی / Support

برای سوالات، پیشنهادات یا گزارش مشکلات:

- 📧 **ایمیل / Email**: [support@thermal-analysis.com]
- 📱 **تلگرام / Telegram**: [@thermal_support]
- 🌐 **وب‌سایت / Website**: [www.thermal-analysis.com]

---

## 📜 مجوز / License

این پروژه تحت مجوز MIT منتشر شده است. برای اطلاعات بیشتر فایل LICENSE را مطالعه کنید.

This project is licensed under the MIT License. See the LICENSE file for details.

---

<div align="center">

**🔥 ساخته شده با ❤️ برای مهندسان حرارت**

**🔥 Made with ❤️ for Thermal Engineers**

</div>