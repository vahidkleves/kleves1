import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os
import glob
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedThermalAnalyzer:
    """
    Advanced Thermal Insulation Analysis System using Machine Learning and Thermal Physics
    
    This system analyzes thermal insulation performance using data from Kloriz software
    and predicts surface temperatures for complex geometries using advanced ML algorithms
    combined with thermal physics principles.
    """
    
    def __init__(self):
        self.kloriz_data = pd.DataFrame()
        self.insulation_data = pd.DataFrame()
        self.combined_data = pd.DataFrame()
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Physical constants
        self.STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)
        self.AIR_THERMAL_CONDUCTIVITY = 0.026  # W/(m·K) at 25°C
        
        # Insulation types as specified
        self.insulation_types = [
            'Cerablanket',
            'Silika Needeled Mat', 
            'Rock Wool',
            'Needeled Mat'
        ]
        
        # Equipment geometries
        self.equipment_types = [
            'Horizontal Pipe',
            'Vertical Pipe', 
            'Horizontal Flat Surface',
            'Vertical Flat Surface',
            'Sphere',
            'Cube',
            'Turbine V94.2',
            'Valve',
            'Compressor',
            'Heat Exchanger',
            'Reactor',
            'Tank'
        ]
        
    def parse_html_files(self, html_directory: str = "/workspace") -> pd.DataFrame:
        """
        Parse HTML files from Kloriz software to extract thermal data
        """
        print("🔍 Parsing HTML files from Kloriz software...")
        
        html_files = glob.glob(os.path.join(html_directory, "a*.html"))
        html_files.sort()
        
        data_list = []
        
        for file_path in html_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file.read(), 'html.parser')
                
                # Extract data using CSS selectors
                equipment_type = soup.select_one('.equipment_type')
                internal_temp = soup.select_one('.internal_temp')
                ambient_temp = soup.select_one('.ambient_temp')
                wind_speed = soup.select_one('.wind_speed')
                thickness = soup.select_one('.thickness')
                surface_area = soup.select_one('.surface_area')
                surface_temp = soup.select_one('.surface_temp')
                
                if all([equipment_type, internal_temp, ambient_temp, wind_speed, 
                       thickness, surface_area, surface_temp]):
                    
                    data_dict = {
                        'file_name': os.path.basename(file_path),
                        'equipment_type': equipment_type.get_text().strip(),
                        'internal_temperature': float(internal_temp.get_text().strip()),
                        'ambient_temperature': float(ambient_temp.get_text().strip()),
                        'wind_speed': float(wind_speed.get_text().strip()),
                        'total_thickness': float(thickness.get_text().strip()),
                        'surface_area': float(surface_area.get_text().strip()),
                        'surface_temperature': float(surface_temp.get_text().strip())
                    }
                    
                    data_list.append(data_dict)
                    print(f"✅ Parsed {os.path.basename(file_path)}: {data_dict['equipment_type']}")
                
            except Exception as e:
                print(f"❌ Error parsing {file_path}: {str(e)}")
        
        self.kloriz_data = pd.DataFrame(data_list)
        print(f"🎯 Successfully parsed {len(self.kloriz_data)} thermal analysis records")
        
        return self.kloriz_data
    
    def load_insulation_data(self, excel_file: str = "/workspace/inputdata.xlsx") -> pd.DataFrame:
        """
        Load insulation properties from Excel file
        """
        print("📊 Loading insulation properties data...")
        
        try:
            self.insulation_data = pd.read_excel(excel_file)
            print(f"✅ Loaded {len(self.insulation_data)} insulation records")
            print(f"📋 Columns: {list(self.insulation_data.columns)}")
            
            # Display insulation types distribution
            if 'Insulation_Type' in self.insulation_data.columns:
                print("\n🔍 Insulation Types Distribution:")
                print(self.insulation_data['Insulation_Type'].value_counts())
            
            return self.insulation_data
            
        except Exception as e:
            print(f"❌ Error loading insulation data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_thermal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced thermal physics features
        """
        print("🧮 Calculating thermal physics features...")
        
        # Temperature difference
        data['temp_difference'] = data['internal_temperature'] - data['ambient_temperature']
        
        # Thermal resistance (simplified)
        data['thermal_resistance'] = data['total_thickness'] / 1000  # Convert mm to m
        
        # Heat transfer coefficient estimation (forced convection)
        data['convection_coefficient'] = 10.45 - data['wind_speed'] + 10 * np.sqrt(data['wind_speed'])
        
        # Radiative heat transfer component
        T_internal_K = data['internal_temperature'] + 273.15
        T_ambient_K = data['ambient_temperature'] + 273.15
        data['radiation_factor'] = self.STEFAN_BOLTZMANN * (T_internal_K**4 - T_ambient_K**4)
        
        # Surface area to volume ratio (geometry factor)
        data['area_factor'] = np.log1p(data['surface_area'])
        
        # Thermal diffusivity estimation
        data['thermal_diffusivity'] = data['temp_difference'] / (data['total_thickness'] + 1)
        
        # Equipment complexity factor
        complexity_map = {
            'Horizontal Pipe': 1.0,
            'Vertical Pipe': 1.1,
            'Horizontal Flat Surface': 0.8,
            'Vertical Flat Surface': 0.9,
            'Sphere': 1.2,
            'Cube': 1.0,
            'Turbine V94.2': 2.5,
            'Valve': 2.0,
            'Compressor': 2.8,
            'Heat Exchanger': 2.2,
            'Reactor': 1.8,
            'Tank': 1.3
        }
        data['complexity_factor'] = data['equipment_type'].map(complexity_map).fillna(1.5)
        
        # Combined thermal index
        data['thermal_index'] = (data['temp_difference'] * data['complexity_factor']) / (data['thermal_resistance'] + 0.001)
        
        print("✅ Thermal features calculated successfully")
        return data
    
    def prepare_training_data(self) -> pd.DataFrame:
        """
        Prepare and combine all data for training
        """
        print("🔧 Preparing training data...")
        
        # Start with Kloriz data
        combined = self.kloriz_data.copy()
        
        # Calculate thermal features
        combined = self.calculate_thermal_features(combined)
        
        # Add insulation properties (simplified approach - using average properties)
        if not self.insulation_data.empty:
            # Calculate average properties for each insulation type
            insulation_avg = self.insulation_data.groupby('Insulation_Type').agg({
                'Thickness_mm': 'mean',
                'Density_kg_m3': 'mean',
                'Thermal_Conductivity_W_mK': 'mean',
                'Convection_Coefficient_W_m2K': 'mean',
                'Max_Temperature_C': 'mean'
            }).reset_index()
            
            # For this implementation, we'll use Rock Wool as default
            # In a real scenario, you'd have insulation type mapping for each equipment
            default_insulation = insulation_avg[insulation_avg['Insulation_Type'] == 'Rock Wool'].iloc[0]
            
            combined['insulation_density'] = default_insulation['Density_kg_m3']
            combined['insulation_conductivity'] = default_insulation['Thermal_Conductivity_W_mK']
            combined['insulation_convection'] = default_insulation['Convection_Coefficient_W_m2K']
            combined['max_temp_rating'] = default_insulation['Max_Temperature_C']
        
        # Encode categorical variables
        le_equipment = LabelEncoder()
        combined['equipment_type_encoded'] = le_equipment.fit_transform(combined['equipment_type'])
        self.label_encoders['equipment_type'] = le_equipment
        
        self.combined_data = combined
        print(f"✅ Training data prepared: {len(combined)} samples with {len(combined.columns)} features")
        
        return combined
    
    def train_models(self) -> Dict:
        """
        Train multiple machine learning models
        """
        print("🤖 Training machine learning models...")
        
        if self.combined_data.empty:
            print("❌ No training data available")
            return {}
        
        # Prepare features and target
        feature_columns = [
            'internal_temperature', 'ambient_temperature', 'wind_speed',
            'total_thickness', 'surface_area', 'temp_difference',
            'thermal_resistance', 'convection_coefficient', 'radiation_factor',
            'area_factor', 'thermal_diffusivity', 'complexity_factor',
            'thermal_index', 'equipment_type_encoded'
        ]
        
        # Add insulation features if available
        if 'insulation_density' in self.combined_data.columns:
            feature_columns.extend([
                'insulation_density', 'insulation_conductivity',
                'insulation_convection', 'max_temp_rating'
            ])
        
        X = self.combined_data[feature_columns]
        y = self.combined_data['surface_temperature']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            ),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=1000,
                random_state=42
            )
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"🔄 Training {name}...")
            
            if name == 'Neural Network':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"✅ {name} - R²: {r2:.4f}, RMSE: {rmse:.2f}°C, MAE: {mae:.2f}°C")
        
        # Select best model based on R² score
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        self.best_model = results[best_model_name]['model']
        self.models = results
        
        print(f"🏆 Best model: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})")
        
        return results
    
    def predict_temperature(self, equipment_data: Dict) -> Dict:
        """
        Predict surface temperature for new equipment
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Please train models first.")
        
        print(f"🔮 Predicting surface temperature for {equipment_data.get('equipment_type', 'Unknown')}...")
        
        # Create DataFrame from input
        input_df = pd.DataFrame([equipment_data])
        
        # Calculate thermal features
        input_df = self.calculate_thermal_features(input_df)
        
        # Add insulation properties (using defaults)
        if not self.insulation_data.empty:
            default_insulation = self.insulation_data[
                self.insulation_data['Insulation_Type'] == 'Rock Wool'
            ].iloc[0] if len(self.insulation_data[self.insulation_data['Insulation_Type'] == 'Rock Wool']) > 0 else self.insulation_data.iloc[0]
            
            input_df['insulation_density'] = default_insulation['Density_kg_m3']
            input_df['insulation_conductivity'] = default_insulation['Thermal_Conductivity_W_mK']
            input_df['insulation_convection'] = default_insulation['Convection_Coefficient_W_m2K']
            input_df['max_temp_rating'] = default_insulation['Max_Temperature_C']
        
        # Encode equipment type
        try:
            input_df['equipment_type_encoded'] = self.label_encoders['equipment_type'].transform([equipment_data['equipment_type']])[0]
        except:
            # If equipment type not seen before, use average encoding
            input_df['equipment_type_encoded'] = len(self.label_encoders['equipment_type'].classes_) // 2
        
        # Prepare features
        feature_columns = [
            'internal_temperature', 'ambient_temperature', 'wind_speed',
            'total_thickness', 'surface_area', 'temp_difference',
            'thermal_resistance', 'convection_coefficient', 'radiation_factor',
            'area_factor', 'thermal_diffusivity', 'complexity_factor',
            'thermal_index', 'equipment_type_encoded'
        ]
        
        if 'insulation_density' in input_df.columns:
            feature_columns.extend([
                'insulation_density', 'insulation_conductivity',
                'insulation_convection', 'max_temp_rating'
            ])
        
        X_pred = input_df[feature_columns]
        
        # Scale if using Neural Network
        if isinstance(self.best_model, MLPRegressor):
            X_pred = self.scaler.transform(X_pred)
        
        # Make prediction
        predicted_temp = self.best_model.predict(X_pred)[0]
        
        # Calculate confidence interval (simplified)
        confidence_range = abs(predicted_temp * 0.05)  # ±5% confidence
        
        result = {
            'predicted_surface_temperature': round(predicted_temp, 2),
            'confidence_range': round(confidence_range, 2),
            'equipment_type': equipment_data['equipment_type'],
            'model_used': type(self.best_model).__name__
        }
        
        print(f"🎯 Predicted surface temperature: {result['predicted_surface_temperature']}°C ± {result['confidence_range']}°C")
        
        return result
    
    def save_model(self, filepath: str = "/workspace/advanced_thermal_model.joblib"):
        """
        Save the trained model
        """
        model_data = {
            'best_model': self.best_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'models': self.models
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str = "/workspace/advanced_thermal_model.joblib"):
        """
        Load a trained model
        """
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data['best_model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.models = model_data.get('models', {})
            print(f"📂 Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def generate_analysis_report(self, output_file: str = "/workspace/advanced_thermal_report.html"):
        """
        Generate comprehensive analysis report
        """
        print("📋 Generating analysis report...")
        
        html_content = f"""
        <!DOCTYPE html>
        <html dir="rtl" lang="fa">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>گزارش تحلیل پیشرفته حرارتی</title>
            <style>
                body {{ font-family: 'Tahoma', Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; border-radius: 10px; }}
                .content {{ padding: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #007bff; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .value {{ font-weight: bold; color: #007bff; font-size: 1.2em; }}
                .model-comparison {{ display: flex; justify-content: space-around; flex-wrap: wrap; }}
                .model-card {{ background: white; padding: 15px; margin: 10px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); min-width: 200px; }}
                .best-model {{ border: 3px solid #28a745; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔬 سیستم تحلیل پیشرفته حرارتی</h1>
                <p>تحلیل هوشمند عایق‌کاری با یادگیری ماشین</p>
            </div>
            
            <div class="content">
                <div class="section">
                    <h2>📊 خلاصه داده‌ها</h2>
                    <div class="metric">
                        <div>تعداد نمونه‌های آموزشی</div>
                        <div class="value">{len(self.kloriz_data)} نمونه</div>
                    </div>
                    <div class="metric">
                        <div>انواع تجهیزات</div>
                        <div class="value">{len(self.kloriz_data['equipment_type'].unique()) if not self.kloriz_data.empty else 0} نوع</div>
                    </div>
                    <div class="metric">
                        <div>انواع عایق</div>
                        <div class="value">{len(self.insulation_data['Insulation_Type'].unique()) if not self.insulation_data.empty else 0} نوع</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🤖 مقایسه مدل‌های یادگیری ماشین</h2>
                    <div class="model-comparison">
        """
        
        # Add model comparison
        if self.models:
            for name, results in self.models.items():
                is_best = (results['model'] == self.best_model)
                card_class = "model-card best-model" if is_best else "model-card"
                
                html_content += f"""
                        <div class="{card_class}">
                            <h3>{'🏆 ' if is_best else ''}{name}</h3>
                            <p><strong>R² Score:</strong> {results['r2']:.4f}</p>
                            <p><strong>RMSE:</strong> {results['rmse']:.2f}°C</p>
                            <p><strong>MAE:</strong> {results['mae']:.2f}°C</p>
                            {'<p style="color: #28a745;"><strong>بهترین مدل</strong></p>' if is_best else ''}
                        </div>
                """
        
        html_content += f"""
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔍 ویژگی‌های محاسبه شده</h2>
                    <ul>
                        <li>اختلاف دما (Temperature Difference)</li>
                        <li>مقاومت حرارتی (Thermal Resistance)</li>
                        <li>ضریب انتقال حرارت جابجایی (Convection Coefficient)</li>
                        <li>فاکتور تشعشع (Radiation Factor)</li>
                        <li>فاکتور پیچیدگی هندسی (Complexity Factor)</li>
                        <li>شاخص حرارتی ترکیبی (Thermal Index)</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>📈 نتایج تحلیل</h2>
                    <p>سیستم با استفاده از ترکیب اصول فیزیک حرارت و الگوریتم‌های یادگیری ماشین، قادر به پیش‌بینی دمای سطح تجهیزات پیچیده است.</p>
                    <p><strong>دقت مدل بهینه:</strong> <span class="value">{max(results['r2'] for results in self.models.values()):.1%}</span></p>
                </div>
                
                <div style="margin-top: 30px; text-align: center; color: #6c757d;">
                    <p>تاریخ تحلیل: {pd.Timestamp.now().strftime('%Y/%m/%d - %H:%M')}</p>
                    <p>سیستم تحلیل پیشرفته حرارتی - نسخه 3.0</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📋 Report generated: {output_file}")
        return output_file

def main():
    """
    Main execution function
    """
    print("🚀 Starting Advanced Thermal Insulation Analysis System")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = AdvancedThermalAnalyzer()
    
    # Parse HTML data
    analyzer.parse_html_files()
    
    # Load insulation data
    analyzer.load_insulation_data()
    
    # Prepare training data
    analyzer.prepare_training_data()
    
    # Train models
    analyzer.train_models()
    
    # Save model
    analyzer.save_model()
    
    # Generate report
    analyzer.generate_analysis_report()
    
    print("\n🎉 Analysis completed successfully!")
    print("=" * 60)
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()