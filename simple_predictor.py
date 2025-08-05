#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Thermal Predictor - Easy Interface
=========================================

Simple interface for predicting surface temperatures of thermal equipment
using the trained machine learning model.

Usage:
    python3 simple_predictor.py

Author: Advanced Thermal Analysis System
Version: 3.0
"""

from advanced_thermal_analyzer import AdvancedThermalAnalyzer
import pandas as pd

def predict_single_equipment():
    """
    Interactive prediction for a single piece of equipment
    """
    print("🔮 THERMAL SURFACE TEMPERATURE PREDICTOR")
    print("=" * 50)
    print("لطفاً مشخصات تجهیز خود را وارد کنید:")
    print()
    
    # Load the trained model
    analyzer = AdvancedThermalAnalyzer()
    if not analyzer.load_model("/workspace/advanced_thermal_model.joblib"):
        print("❌ Model not found. Please run the training first.")
        return
    
    # Load insulation data
    analyzer.load_insulation_data()
    
    # Get user input
    try:
        print("انواع تجهیزات موجود:")
        equipment_types = [
            'Horizontal Pipe', 'Vertical Pipe', 'Horizontal Flat Surface',
            'Vertical Flat Surface', 'Sphere', 'Cube', 'Turbine V94.2',
            'Valve', 'Compressor', 'Heat Exchanger', 'Reactor', 'Tank'
        ]
        
        for i, eq_type in enumerate(equipment_types, 1):
            print(f"  {i}. {eq_type}")
        
        choice = int(input("\nانتخاب نوع تجهیز (شماره): ")) - 1
        equipment_type = equipment_types[choice] if 0 <= choice < len(equipment_types) else 'Compressor'
        
        internal_temp = float(input("دمای داخلی تجهیز (درجه سانتیگراد): "))
        ambient_temp = float(input("دمای محیط (درجه سانتیگراد): "))
        wind_speed = float(input("سرعت باد (متر بر ثانیه): "))
        thickness = float(input("ضخامت کل عایق (میلی‌متر): "))
        surface_area = float(input("مساحت کل سطح (متر مربع): "))
        
        equipment_data = {
            'equipment_type': equipment_type,
            'internal_temperature': internal_temp,
            'ambient_temperature': ambient_temp,
            'wind_speed': wind_speed,
            'total_thickness': thickness,
            'surface_area': surface_area
        }
        
        print("\n🔄 در حال محاسبه...")
        
        # Make prediction
        result = analyzer.predict_temperature(equipment_data)
        
        # Display results
        print("\n" + "=" * 50)
        print("🎯 نتایج پیش‌بینی:")
        print("=" * 50)
        print(f"نوع تجهیز: {equipment_type}")
        print(f"دمای سطح پیش‌بینی شده: {result['predicted_surface_temperature']}°C")
        print(f"محدوده اطمینان: ±{result['confidence_range']}°C")
        print(f"مدل استفاده شده: {result['model_used']}")
        
        # Safety analysis
        surface_temp = result['predicted_surface_temperature']
        if surface_temp <= 60:
            safety = "ایمن ✅"
            recommendation = "تجهیز برای عملیات عادی و تماس پرسنل ایمن است."
        elif surface_temp <= 80:
            safety = "احتیاط ⚠️"
            recommendation = "هنگام کار با تجهیز احتیاط کنید. نصب علائم هشدار را در نظر بگیرید."
        elif surface_temp <= 100:
            safety = "هشدار ⚠️"
            recommendation = "علائم هشدار و موانع نصب کنید. دسترسی پرسنل را محدود کنید."
        else:
            safety = "خطرناک ❌"
            recommendation = "بحرانی: موانع محافظ و سیستم‌های هشدار نصب کنید. دسترسی را محدود کنید."
        
        print(f"\n🛡️ تحلیل ایمنی:")
        print(f"سطح ایمنی: {safety}")
        print(f"توصیه: {recommendation}")
        
        # Save result
        result_df = pd.DataFrame([{
            'Equipment_Type': equipment_type,
            'Internal_Temp_C': internal_temp,
            'Ambient_Temp_C': ambient_temp,
            'Wind_Speed_ms': wind_speed,
            'Thickness_mm': thickness,
            'Surface_Area_m2': surface_area,
            'Predicted_Surface_Temp_C': result['predicted_surface_temperature'],
            'Confidence_Range_C': result['confidence_range'],
            'Safety_Level': safety,
            'Recommendation': recommendation
        }])
        
        result_df.to_excel('/workspace/prediction_result.xlsx', index=False)
        print(f"\n💾 نتایج در فایل ذخیره شد: /workspace/prediction_result.xlsx")
        
    except Exception as e:
        print(f"❌ خطا در پردازش: {str(e)}")

def quick_prediction_examples():
    """
    Quick prediction examples for common equipment
    """
    print("\n🚀 QUICK PREDICTION EXAMPLES")
    print("=" * 50)
    
    analyzer = AdvancedThermalAnalyzer()
    if not analyzer.load_model("/workspace/advanced_thermal_model.joblib"):
        print("❌ Model not found. Please run the training first.")
        return
    
    analyzer.load_insulation_data()
    
    examples = [
        {
            'name': 'توربین گازی صنعتی',
            'equipment_type': 'Turbine V94.2',
            'internal_temperature': 550.0,
            'ambient_temperature': 25.0,
            'wind_speed': 2.0,
            'total_thickness': 120.0,
            'surface_area': 40.0
        },
        {
            'name': 'کمپرسور مرکزگریز',
            'equipment_type': 'Compressor',
            'internal_temperature': 350.0,
            'ambient_temperature': 30.0,
            'wind_speed': 1.5,
            'total_thickness': 100.0,
            'surface_area': 18.0
        },
        {
            'name': 'مبدل حرارتی',
            'equipment_type': 'Heat Exchanger',
            'internal_temperature': 400.0,
            'ambient_temperature': 28.0,
            'wind_speed': 2.5,
            'total_thickness': 90.0,
            'surface_area': 25.0
        }
    ]
    
    for example in examples:
        print(f"\n🔧 {example['name']}:")
        result = analyzer.predict_temperature(example)
        print(f"   دمای سطح: {result['predicted_surface_temperature']}°C ± {result['confidence_range']}°C")

def main():
    """
    Main function
    """
    print("🌡️ سیستم پیش‌بینی دمای سطح عایق حرارتی")
    print("=" * 60)
    print("این سیستم با استفاده از داده‌های نرم‌افزار کلورز و یادگیری ماشین")
    print("دمای سطح تجهیزات پیچیده را پیش‌بینی می‌کند")
    print("=" * 60)
    
    while True:
        print("\nانتخاب کنید:")
        print("1. پیش‌بینی برای یک تجهیز")
        print("2. مثال‌های سریع")
        print("3. خروج")
        
        try:
            choice = input("\nانتخاب شما (1-3): ").strip()
            
            if choice == '1':
                predict_single_equipment()
            elif choice == '2':
                quick_prediction_examples()
            elif choice == '3':
                print("خداحافظ! 👋")
                break
            else:
                print("انتخاب نامعتبر. لطفاً دوباره تلاش کنید.")
                
        except KeyboardInterrupt:
            print("\n\nخداحافظ! 👋")
            break
        except Exception as e:
            print(f"خطا: {str(e)}")

if __name__ == "__main__":
    main()