# 🔥 سیستم تحلیل حرارتی عایق‌ها با یادگیری ماشین
# Thermal Insulation Analysis System with Machine Learning

یک سیستم پیشرفته برای تحلیل عملکرد عایق‌های حرارتی با استفاده از یادگیری ماشین که قادر به پیش‌بینی دمای سطح تجهیزات پیچیده است.

An advanced system for analyzing thermal insulation performance using machine learning, capable of predicting surface temperatures for complex equipment geometries.

## 📋 ویژگی‌ها / Features

### 🎯 قابلیت‌های اصلی / Main Capabilities
- **پارس کردن فایل‌های HTML خروجی نرم‌افزار کلورز** / Parse Kloriz software HTML output files
- **خواندن داده‌های عایق از فایل Excel** / Read insulation data from Excel files  
- **یادگیری ماشین برای پیش‌بینی دمای سطح** / Machine learning for surface temperature prediction
- **پشتیبانی از انواع مختلف عایق** / Support for various insulation types
- **تحلیل تجهیزات پیچیده** / Analysis of complex equipment geometries

### 🧱 انواع عایق پشتیبانی شده / Supported Insulation Types
- **Cerablanket** - عایق سرامیکی با مقاومت حرارتی بالا
- **Silika Needeled Mat** - پشم سیلیکا سوزنی شده  
- **Rock Wool** - پشم سنگ معدنی
- **Needeled Mat** - پشم سوزنی شده

### 🔧 انواع تجهیزات / Equipment Types
- لوله افقی و عمودی / Horizontal & Vertical Pipes
- سطح صاف افقی و عمودی / Horizontal & Vertical Flat Surfaces  
- کره و مکعب / Sphere & Cube
- توربین V94.2 / Turbine V94.2
- ولوها / Valves
- کمپرسور / Compressors
- توربین‌های پیچیده / Complex Turbines
- تجهیزات پیچیده / Complex Equipment

## 🚀 نصب و راه‌اندازی / Installation & Setup

### پیش‌نیازها / Prerequisites
```bash
Python 3.8+
pip
```

### 1. نصب وابستگی‌ها / Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. ایجاد داده‌های نمونه / Create Sample Data
```bash
python create_sample_data.py
```

### 3. اجرای تحلیل اصلی / Run Main Analysis
```bash
python thermal_insulation_analyzer.py
```

### 4. اجرای مثال‌های کاربردی / Run Usage Examples
```bash
python usage_example.py
```

## 📁 ساختار فایل‌ها / File Structure

```
workspace/
├── thermal_insulation_analyzer.py    # کلاس اصلی سیستم / Main system class
├── create_sample_data.py             # تولید داده‌های نمونه / Sample data generator
├── usage_example.py                  # مثال‌های کاربردی / Usage examples
├── requirements.txt                  # وابستگی‌ها / Dependencies
├── README.md                         # مستندات / Documentation
├── inputdata.xlsx                    # داده‌های عایق / Insulation data
├── a1.html, a2.html, ...            # فایل‌های خروجی کلورز / Kloriz output files
└── Generated Files:
    ├── thermal_model.joblib          # مدل آموزش دیده / Trained model
    ├── thermal_analysis_report.html # گزارش تحلیل / Analysis report
    ├── batch_predictions.xlsx       # نتایج پیش‌بینی دسته‌ای / Batch predictions
    └── insulation_comparison.xlsx    # مقایسه عایق‌ها / Insulation comparison
```

## 💻 نحوه استفاده / Usage

### استفاده پایه / Basic Usage

```python
from thermal_insulation_analyzer import ThermalInsulationAnalyzer

# ایجاد نمونه از تحلیلگر / Create analyzer instance
analyzer = ThermalInsulationAnalyzer()

# بارگذاری داده‌ها / Load data
analyzer.parse_kloriz_html_files("a*.html")
analyzer.load_insulation_data("inputdata.xlsx")

# آموزش مدل / Train model
analyzer.train_model()

# پیش‌بینی برای تجهیز جدید / Predict for new equipment
equipment_data = {
    'internal_temperature': 450.0,      # °C
    'ambient_temperature': 25.0,        # °C
    'wind_speed': 3.5,                  # m/s
    'total_insulation_thickness': 80.0, # mm
    'total_surface_area': 15.0,         # m²
    'average_density': 120.0,           # kg/m³
    'average_thermal_conductivity': 0.042, # W/m·K
    'average_convection_coefficient': 15.0, # W/m²·K
    'number_of_layers': 2,
    'equipment_type': 'Complex Turbine',
    'dominant_insulation_type': 'Rock Wool'
}

result = analyzer.predict_surface_temperature(equipment_data)
print(f"دمای سطح پیش‌بینی شده: {result['predicted_surface_temperature']} °C")
```

### پیش‌بینی دسته‌ای / Batch Prediction

```python
# لیست تجهیزات مختلف / List of different equipment
equipment_list = [
    {
        'name': 'Compressor Unit A',
        'internal_temperature': 350.0,
        'ambient_temperature': 25.0,
        # ... سایر پارامترها / other parameters
    },
    # ... تجهیزات بیشتر / more equipment
]

# پیش‌بینی برای همه / Predict for all
for equipment in equipment_list:
    result = analyzer.predict_surface_temperature(equipment)
    print(f"{equipment['name']}: {result['predicted_surface_temperature']} °C")
```

## 🔍 الگوریتم‌های یادگیری ماشین / Machine Learning Algorithms

سیستم از الگوریتم‌های زیر استفاده می‌کند و بهترین را انتخاب می‌کند:

The system uses the following algorithms and selects the best one:

### 1. Random Forest Regressor
- مقاوم در برابر overfitting / Robust against overfitting
- قابلیت تحلیل اهمیت ویژگی‌ها / Feature importance analysis
- عملکرد مناسب با داده‌های پیچیده / Good performance with complex data

### 2. Gradient Boosting Regressor  
- دقت بالا در پیش‌بینی / High prediction accuracy
- قابلیت یادگیری الگوهای پیچیده / Learning complex patterns
- بهینه‌سازی تدریجی / Sequential optimization

### ارزیابی مدل / Model Evaluation
- **R² Score**: ضریب تعیین / Coefficient of determination
- **RMSE**: جذر میانگین مربعات خطا / Root Mean Square Error  
- **MAE**: میانگین قدر مطلق خطا / Mean Absolute Error
- **Cross-validation**: اعتبارسنجی متقابل 5-fold

## 📊 فرمت داده‌های ورودی / Input Data Format

### فایل‌های HTML کلورز / Kloriz HTML Files
فایل‌های HTML باید حاوی اطلاعات زیر باشند:

HTML files should contain the following information:

```html
<span class="value internal_temp">450.0</span>     <!-- دمای داخلی / Internal temperature -->
<span class="value ambient_temp">25.0</span>       <!-- دمای محیط / Ambient temperature -->
<span class="value wind_speed">3.5</span>          <!-- سرعت باد / Wind speed -->
<span class="value thickness">80.0</span>          <!-- ضخامت عایق / Insulation thickness -->
<span class="value surface_area">15.0</span>       <!-- مساحت سطح / Surface area -->
<span class="value surface_temp">65.2</span>       <!-- دمای سطح / Surface temperature -->
<span class="value equipment_type">Turbine</span>  <!-- نوع تجهیز / Equipment type -->
```

### فایل Excel عایق‌ها / Insulation Excel File
فایل Excel باید دارای ستون‌های زیر باشد:

Excel file should have the following columns:

| Column | Description | واحد / Unit |
|--------|-------------|-------------|
| `Insulation_Type` | نوع عایق / Insulation type | - |
| `Thickness_mm` | ضخامت / Thickness | mm |
| `Density_kg_m3` | چگالی / Density | kg/m³ |
| `Thermal_Conductivity_W_mK` | ضریب انتقال حرارت / Thermal conductivity | W/m·K |
| `Convection_Coefficient_W_m2K` | ضریب جابجایی / Convection coefficient | W/m²·K |

## 🎯 دقت و عملکرد / Accuracy & Performance

### معیارهای عملکرد معمول / Typical Performance Metrics
- **R² Score**: > 0.85
- **RMSE**: < 10°C  
- **MAE**: < 7°C
- **Training Time**: < 2 minutes
- **Prediction Time**: < 1 second per sample

### فاکتورهای مؤثر بر دقت / Factors Affecting Accuracy
- **کیفیت داده‌های آموزشی** / Training data quality
- **تنوع انواع تجهیزات** / Equipment type diversity  
- **تعداد نمونه‌های آموزشی** / Number of training samples
- **کالیبراسیون پارامترها** / Parameter calibration

## 🔧 تنظیمات پیشرفته / Advanced Configuration

### تنظیم پارامترهای مدل / Model Parameter Tuning

```python
# تنظیم Random Forest
analyzer.train_model(
    test_size=0.2,
    random_state=42,
    n_estimators=200,      # تعداد درخت‌ها / Number of trees
    max_depth=15,          # عمق درخت‌ها / Tree depth
    min_samples_split=5    # حداقل نمونه برای تقسیم / Min samples for split
)
```

### اضافه کردن ویژگی‌های جدید / Adding New Features

```python
# اضافه کردن ویژگی‌های سفارشی / Add custom features
analyzer.feature_columns.extend([
    'custom_parameter_1',
    'custom_parameter_2'
])
```

## 🐛 عیب‌یابی / Troubleshooting

### مشکلات رایج / Common Issues

#### 1. خطای پارس کردن HTML / HTML Parsing Error
```
❌ Error parsing a1.html: ...
```
**راه حل / Solution**: بررسی فرمت فایل HTML و اطمینان از وجود تگ‌های مورد نیاز

#### 2. خطای بارگذاری Excel / Excel Loading Error  
```
❌ Error loading Excel file: ...
```
**راه حل / Solution**: بررسی فرمت فایل Excel و نام ستون‌ها

#### 3. خطای آموزش مدل / Model Training Error
```
❌ Error training model: ...
```
**راه حل / Solution**: بررسی کفایت داده‌های آموزشی و حذف مقادیر نامعتبر

### فعال‌سازی حالت دیباگ / Enable Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

analyzer = ThermalInsulationAnalyzer()
analyzer.debug_mode = True
```

## 📈 بهبود عملکرد / Performance Optimization

### 1. بهینه‌سازی داده‌ها / Data Optimization
- حذف داده‌های پرت / Remove outliers
- نرمال‌سازی ویژگی‌ها / Feature normalization  
- انتخاب ویژگی‌های مهم / Feature selection

### 2. بهینه‌سازی مدل / Model Optimization
- تنظیم hyperparameter / Hyperparameter tuning
- استفاده از ensemble methods / Use ensemble methods
- کراس ولیدیشن / Cross-validation

### 3. بهینه‌سازی حافظه / Memory Optimization
- پردازش داده‌ها به صورت batch / Batch processing
- ذخیره مدل‌های فشرده / Compressed model storage

## 🤝 مشارکت / Contributing

برای مشارکت در پروژه:

To contribute to the project:

1. Fork کردن پروژه / Fork the project
2. ایجاد branch جدید / Create a new branch
3. اعمال تغییرات / Make changes  
4. ارسال Pull Request / Submit Pull Request

## 📄 مجوز / License

این پروژه تحت مجوز MIT منتشر شده است.

This project is released under the MIT License.

## 📞 پشتیبانی / Support

برای سؤالات و پشتیبانی:

For questions and support:

- **Issues**: GitHub Issues
- **Documentation**: این فایل README / This README file
- **Examples**: فایل `usage_example.py` / `usage_example.py` file

## 🔄 نسخه‌ها / Versions

### v1.0.0 (Current)
- پیاده‌سازی اولیه سیستم / Initial system implementation
- پشتیبانی از 4 نوع عایق / Support for 4 insulation types
- الگوریتم‌های Random Forest و Gradient Boosting / Random Forest & Gradient Boosting algorithms
- پارسر HTML برای کلورز / HTML parser for Kloriz
- خواننده Excel برای داده‌های عایق / Excel reader for insulation data

### آپدیت‌های آینده / Future Updates
- پشتیبانی از انواع عایق بیشتر / Support for more insulation types
- بهبود دقت مدل‌ها / Improved model accuracy
- رابط کاربری گرافیکی / Graphical user interface
- API برای استفاده از راه دور / Remote API access

---

**ساخته شده با ❤️ برای تحلیل حرارتی دقیق / Made with ❤️ for accurate thermal analysis**