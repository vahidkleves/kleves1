import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os
import glob
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ThermalInsulationAnalyzer:
    """
    Thermal Insulation Analysis System using Machine Learning
    
    This system analyzes thermal insulation performance using data from Kloriz software
    and predicts surface temperatures for complex geometries using machine learning.
    """
    
    def __init__(self):
        self.kloriz_data = pd.DataFrame()
        self.insulation_data = pd.DataFrame()
        self.combined_data = pd.DataFrame()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
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
            'Complex Turbine',
            'Complex Equipment'
        ]
    
    def parse_kloriz_html_files(self, file_pattern: str = "a*.html") -> pd.DataFrame:
        """
        Parse HTML output files from Kloriz software (a1.html, a2.html, ...)
        
        Args:
            file_pattern: Pattern to match HTML files
            
        Returns:
            DataFrame with parsed data
        """
        print("📁 Parsing Kloriz HTML files...")
        
        html_files = glob.glob(file_pattern)
        if not html_files:
            print(f"⚠️  No HTML files found matching pattern: {file_pattern}")
            return pd.DataFrame()
        
        all_data = []
        
        for file_path in html_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                
                # Extract data from HTML (this is a template - adjust based on actual HTML structure)
                data_row = self._extract_data_from_html(soup, file_path)
                if data_row:
                    all_data.append(data_row)
                    
            except Exception as e:
                print(f"❌ Error parsing {file_path}: {str(e)}")
        
        if all_data:
            self.kloriz_data = pd.DataFrame(all_data)
            print(f"✅ Successfully parsed {len(all_data)} HTML files")
            return self.kloriz_data
        else:
            print("❌ No data extracted from HTML files")
            return pd.DataFrame()
    
    def _extract_data_from_html(self, soup: BeautifulSoup, file_path: str) -> Optional[Dict]:
        """
        Extract thermal data from HTML soup object
        
        Args:
            soup: BeautifulSoup object
            file_path: Path to the HTML file
            
        Returns:
            Dictionary with extracted data
        """
        try:
            # This is a template - adjust selectors based on actual HTML structure
            data = {
                'file_name': os.path.basename(file_path),
                'wind_speed': self._extract_numeric_value(soup, 'wind_speed', 'سرعت باد'),
                'equipment_type': self._extract_text_value(soup, 'equipment_type', 'نوع تجهیز'),
                'internal_temperature': self._extract_numeric_value(soup, 'internal_temp', 'دمای داخلی'),
                'ambient_temperature': self._extract_numeric_value(soup, 'ambient_temp', 'دمای محیط'),
                'total_insulation_thickness': self._extract_numeric_value(soup, 'thickness', 'ضخامت'),
                'total_surface_area': self._extract_numeric_value(soup, 'surface_area', 'مساحت'),
                'surface_temperature': self._extract_numeric_value(soup, 'surface_temp', 'دمای سطح')
            }
            
            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}
            
            return data if len(data) > 3 else None  # Ensure we have meaningful data
            
        except Exception as e:
            print(f"❌ Error extracting data from {file_path}: {str(e)}")
            return None
    
    def _extract_numeric_value(self, soup: BeautifulSoup, class_name: str, persian_keyword: str) -> Optional[float]:
        """Extract numeric value from HTML"""
        try:
            # Try by class name
            element = soup.find(class_=class_name)
            if element:
                text = element.get_text().strip()
                return self._parse_number(text)
            
            # Try by Persian keyword
            elements = soup.find_all(text=lambda text: text and persian_keyword in text)
            for element in elements:
                parent = element.parent
                if parent:
                    text = parent.get_text().strip()
                    number = self._parse_number(text)
                    if number is not None:
                        return number
            
            return None
        except:
            return None
    
    def _extract_text_value(self, soup: BeautifulSoup, class_name: str, persian_keyword: str) -> Optional[str]:
        """Extract text value from HTML"""
        try:
            # Try by class name
            element = soup.find(class_=class_name)
            if element:
                return element.get_text().strip()
            
            # Try by Persian keyword
            elements = soup.find_all(text=lambda text: text and persian_keyword in text)
            for element in elements:
                parent = element.parent
                if parent:
                    return parent.get_text().strip()
            
            return None
        except:
            return None
    
    def _parse_number(self, text: str) -> Optional[float]:
        """Parse number from Persian/English text"""
        try:
            # Remove Persian digits and convert to English
            persian_digits = '۰۱۲۳۴۵۶۷۸۹'
            english_digits = '0123456789'
            
            for persian, english in zip(persian_digits, english_digits):
                text = text.replace(persian, english)
            
            # Extract number using regex
            import re
            numbers = re.findall(r'[-+]?\d*\.?\d+', text)
            if numbers:
                return float(numbers[0])
            return None
        except:
            return None
    
    def load_insulation_data(self, excel_file: str = "inputdata.xlsx") -> pd.DataFrame:
        """
        Load insulation properties from Excel file
        
        Args:
            excel_file: Path to Excel file with insulation data
            
        Returns:
            DataFrame with insulation properties
        """
        print(f"📊 Loading insulation data from {excel_file}...")
        
        try:
            # Try different sheet names
            sheet_names = ['Sheet1', 'Data', 'Insulation', 'inputdata']
            
            for sheet_name in sheet_names:
                try:
                    self.insulation_data = pd.read_excel(excel_file, sheet_name=sheet_name)
                    break
                except:
                    continue
            
            if self.insulation_data.empty:
                self.insulation_data = pd.read_excel(excel_file)
            
            print(f"✅ Successfully loaded insulation data: {self.insulation_data.shape}")
            print(f"📋 Columns: {list(self.insulation_data.columns)}")
            
            return self.insulation_data
            
        except Exception as e:
            print(f"❌ Error loading Excel file: {str(e)}")
            print("🔄 Creating sample insulation data...")
            return self._create_sample_insulation_data()
    
    def _create_sample_insulation_data(self) -> pd.DataFrame:
        """Create sample insulation data if Excel file is not available"""
        
        sample_data = []
        
        # Sample data for different insulation types
        insulation_properties = {
            'Cerablanket': {'density': 128, 'thermal_conductivity': 0.045},
            'Silika Needeled Mat': {'density': 96, 'thermal_conductivity': 0.038},
            'Rock Wool': {'density': 140, 'thermal_conductivity': 0.042},
            'Needeled Mat': {'density': 120, 'thermal_conductivity': 0.040}
        }
        
        for i in range(50):
            insulation_type = np.random.choice(self.insulation_types)
            props = insulation_properties[insulation_type]
            
            sample_data.append({
                'Layer_ID': i + 1,
                'Insulation_Type': insulation_type,
                'Thickness_mm': np.random.uniform(20, 100),  # mm
                'Density_kg_m3': props['density'] * (1 + np.random.uniform(-0.1, 0.1)),
                'Thermal_Conductivity_W_mK': props['thermal_conductivity'] * (1 + np.random.uniform(-0.1, 0.1)),
                'Convection_Coefficient_W_m2K': np.random.uniform(5, 25)
            })
        
        self.insulation_data = pd.DataFrame(sample_data)
        print("✅ Created sample insulation data")
        return self.insulation_data
    
    def combine_data(self) -> pd.DataFrame:
        """
        Combine Kloriz data with insulation properties
        
        Returns:
            Combined DataFrame ready for ML training
        """
        print("🔗 Combining Kloriz and insulation data...")
        
        if self.kloriz_data.empty:
            print("⚠️  No Kloriz data available, creating sample data...")
            self._create_sample_kloriz_data()
        
        if self.insulation_data.empty:
            print("⚠️  No insulation data available, creating sample data...")
            self._create_sample_insulation_data()
        
        # Create combined dataset
        combined_rows = []
        
        for _, kloriz_row in self.kloriz_data.iterrows():
            # Sample insulation layers for this equipment
            n_layers = np.random.randint(1, 4)  # 1-3 layers
            insulation_sample = self.insulation_data.sample(n_layers)
            
            # Aggregate insulation properties
            total_thickness = insulation_sample['Thickness_mm'].sum()
            avg_density = insulation_sample['Density_kg_m3'].mean()
            avg_thermal_conductivity = insulation_sample['Thermal_Conductivity_W_mK'].mean()
            avg_convection_coeff = insulation_sample['Convection_Coefficient_W_m2K'].mean()
            
            # Combine with Kloriz data
            combined_row = kloriz_row.copy()
            combined_row['total_insulation_thickness_calc'] = total_thickness
            combined_row['average_density'] = avg_density
            combined_row['average_thermal_conductivity'] = avg_thermal_conductivity
            combined_row['average_convection_coefficient'] = avg_convection_coeff
            combined_row['number_of_layers'] = n_layers
            combined_row['dominant_insulation_type'] = insulation_sample['Insulation_Type'].mode().iloc[0]
            
            combined_rows.append(combined_row)
        
        self.combined_data = pd.DataFrame(combined_rows)
        print(f"✅ Combined data created: {self.combined_data.shape}")
        print(f"📋 Combined data columns: {list(self.combined_data.columns)}")
        
        return self.combined_data
    
    def _create_sample_kloriz_data(self):
        """Create sample Kloriz data for testing"""
        
        sample_data = []
        
        for i in range(100):
            equipment_type = np.random.choice(self.equipment_types)
            internal_temp = np.random.uniform(200, 600)  # °C
            ambient_temp = np.random.uniform(15, 45)     # °C
            wind_speed = np.random.uniform(0.5, 10)      # m/s
            surface_area = np.random.uniform(1, 50)      # m²
            
            # Calculate approximate surface temperature using heat transfer principles
            delta_t = internal_temp - ambient_temp
            thickness = np.random.uniform(50, 150)  # mm
            
            # Simplified heat transfer calculation
            surface_temp = ambient_temp + delta_t * np.exp(-thickness/100) * (1 + wind_speed/20)
            surface_temp += np.random.normal(0, 5)  # Add noise
            
            sample_data.append({
                'file_name': f'a{i+1}.html',
                'equipment_type': equipment_type,
                'internal_temperature': internal_temp,
                'ambient_temperature': ambient_temp,
                'wind_speed': wind_speed,
                'total_insulation_thickness': thickness,
                'total_surface_area': surface_area,
                'surface_temperature': max(ambient_temp + 5, surface_temp)
            })
        
        self.kloriz_data = pd.DataFrame(sample_data)
        print("✅ Created sample Kloriz data")
    
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning
        
        Returns:
            Tuple of (features, target)
        """
        print("🔧 Preparing features for machine learning...")
        
        if self.combined_data.empty:
            self.combine_data()
        
        # Select features
        feature_columns = [
            'internal_temperature',
            'ambient_temperature', 
            'wind_speed',
            'total_insulation_thickness',
            'total_surface_area',
            'average_density',
            'average_thermal_conductivity',
            'average_convection_coefficient',
            'number_of_layers'
        ]
        
        categorical_columns = ['equipment_type', 'dominant_insulation_type']
        
        # Prepare features
        features = self.combined_data[feature_columns].copy()
        
        # Encode categorical variables
        for col in categorical_columns:
            if col in self.combined_data.columns:
                le = LabelEncoder()
                features[col] = le.fit_transform(self.combined_data[col].astype(str))
                self.label_encoders[col] = le
        
        # Target variable
        target = self.combined_data['surface_temperature']
        
        print(f"✅ Features prepared: {features.shape}")
        print(f"📊 Feature columns: {list(features.columns)}")
        
        return features, target
    
    def train_model(self, test_size: float = 0.2, random_state: int = 42):
        """
        Train machine learning model
        
        Args:
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
        """
        print("🤖 Training machine learning model...")
        
        # Prepare data
        X, y = self.prepare_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try different algorithms
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        }
        
        best_model = None
        best_score = -np.inf
        best_name = ""
        
        for name, model in models.items():
            print(f"🔄 Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            mean_cv_score = cv_scores.mean()
            
            print(f"📊 {name} - CV R² Score: {mean_cv_score:.4f} (±{cv_scores.std()*2:.4f})")
            
            if mean_cv_score > best_score:
                best_score = mean_cv_score
                best_model = model
                best_name = name
        
        self.model = best_model
        
        # Final evaluation
        y_pred = self.model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n🏆 Best Model: {best_name}")
        print(f"📊 Test Results:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   RMSE: {rmse:.2f} °C")
        print(f"   MAE: {mae:.2f} °C")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 Top 5 Most Important Features:")
            for _, row in feature_importance.head().iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return self.model
    
    def predict_surface_temperature(self, equipment_data: Dict) -> Dict:
        """
        Predict surface temperature for new equipment
        
        Args:
            equipment_data: Dictionary with equipment parameters
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained. Please call train_model() first.")
        
        print("🔮 Predicting surface temperature...")
        
        # Prepare input data
        input_data = pd.DataFrame([equipment_data])
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in input_data.columns:
                try:
                    input_data[col] = encoder.transform(input_data[col].astype(str))
                except ValueError:
                    # Handle unknown categories
                    input_data[col] = 0
        
        # Scale features
        input_scaled = self.scaler.transform(input_data)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)[0]
        
        # Calculate confidence interval (approximate)
        if hasattr(self.model, 'estimators_'):
            # For ensemble methods
            predictions = [tree.predict(input_scaled)[0] for tree in self.model.estimators_]
            std_prediction = np.std(predictions)
            confidence_interval = (prediction - 2*std_prediction, prediction + 2*std_prediction)
        else:
            confidence_interval = (prediction - 10, prediction + 10)  # Default ±10°C
        
        result = {
            'predicted_surface_temperature': round(prediction, 2),
            'confidence_interval': (round(confidence_interval[0], 2), round(confidence_interval[1], 2)),
            'input_parameters': equipment_data
        }
        
        print(f"✅ Predicted surface temperature: {result['predicted_surface_temperature']:.2f} °C")
        print(f"📊 Confidence interval: {result['confidence_interval'][0]:.2f} - {result['confidence_interval'][1]:.2f} °C")
        
        return result
    
    def save_model(self, filename: str = "thermal_model.joblib"):
        """Save trained model and preprocessors"""
        if self.model is None:
            raise ValueError("No model to save. Please train a model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        
        joblib.dump(model_data, filename)
        print(f"💾 Model saved to {filename}")
    
    def load_model(self, filename: str = "thermal_model.joblib"):
        """Load trained model and preprocessors"""
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            print(f"📂 Model loaded from {filename}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
    
    def generate_report(self, output_file: str = "thermal_analysis_report.html"):
        """Generate HTML report with analysis results"""
        
        if self.combined_data.empty:
            print("⚠️  No data available for report generation")
            return
        
        html_content = f"""
        <!DOCTYPE html>
        <html dir="rtl" lang="fa">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>گزارش تحلیل حرارتی عایق‌ها</title>
            <style>
                body {{ font-family: 'Tahoma', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1, h2 {{ color: #2c3e50; text-align: center; }}
                .summary {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .stat-box {{ background: #3498db; color: white; padding: 15px; border-radius: 5px; text-align: center; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #34495e; color: white; }}
                .insulation-types {{ background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔥 سیستم تحلیل حرارتی عایق‌ها با یادگیری ماشین</h1>
                
                <div class="summary">
                    <h2>📊 خلاصه داده‌ها</h2>
                    <p><strong>تعداد کل نمونه‌ها:</strong> {len(self.combined_data)}</p>
                    <p><strong>انواع تجهیزات:</strong> {', '.join(self.combined_data['equipment_type'].unique()) if 'equipment_type' in self.combined_data.columns else 'نامشخص'}</p>
                    <p><strong>انواع عایق:</strong> {', '.join(self.insulation_types)}</p>
                </div>
                
                <div class="stats">
                    <div class="stat-box">
                        <h3>میانگین دمای داخلی</h3>
                        <p>{self.combined_data['internal_temperature'].mean():.1f} °C</p>
                    </div>
                    <div class="stat-box">
                        <h3>میانگین دمای سطح</h3>
                        <p>{self.combined_data['surface_temperature'].mean():.1f} °C</p>
                    </div>
                    <div class="stat-box">
                        <h3>میانگین ضخامت عایق</h3>
                        <p>{self.combined_data['total_insulation_thickness'].mean():.1f} mm</p>
                    </div>
                    <div class="stat-box">
                        <h3>میانگین سرعت باد</h3>
                        <p>{self.combined_data['wind_speed'].mean():.1f} m/s</p>
                    </div>
                </div>
                
                <div class="insulation-types">
                    <h2>🧱 مشخصات انواع عایق</h2>
                    <ul>
                        <li><strong>Cerablanket:</strong> عایق سرامیکی با مقاومت حرارتی بالا</li>
                        <li><strong>Silika Needeled Mat:</strong> پشم سیلیکا سوزنی شده</li>
                        <li><strong>Rock Wool:</strong> پشم سنگ معدنی</li>
                        <li><strong>Needeled Mat:</strong> پشم سوزنی شده</li>
                    </ul>
                </div>
                
                <p style="text-align: center; color: #7f8c8d; margin-top: 30px;">
                    تولید شده توسط سیستم تحلیل حرارتی با یادگیری ماشین - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
                </p>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 Report generated: {output_file}")


def main():
    """Main function to demonstrate the thermal insulation analyzer"""
    
    print("🔥 Thermal Insulation Analysis System")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = ThermalInsulationAnalyzer()
    
    # Load data
    analyzer.parse_kloriz_html_files()
    analyzer.load_insulation_data()
    
    # Train model
    analyzer.train_model()
    
    # Example prediction for complex equipment
    example_equipment = {
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
    
    print("\n🔮 Example Prediction:")
    print("-" * 30)
    result = analyzer.predict_surface_temperature(example_equipment)
    
    # Save model
    analyzer.save_model()
    
    # Generate report
    analyzer.generate_report()
    
    print("\n✅ Analysis complete!")
    print("💾 Model saved as 'thermal_model.joblib'")
    print("📄 Report generated as 'thermal_analysis_report.html'")


if __name__ == "__main__":
    main()