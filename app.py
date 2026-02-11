from flask import Flask, render_template, url_for, flash, redirect, request
from flask_login import LoginManager, login_user, current_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import or_, func
import numpy as np
import csv
import os
from datetime import datetime
from config import Config
from models import db, User, Patient
from utils import predict_dr_stage, check_operation_suitability, get_clinical_findings

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Database Creation ---
with app.app_context():
    db.create_all()
    # Create Admin User if not exists
    if not User.query.filter_by(username='sowkya').first():
        hashed_password = generate_password_hash('1234')
        admin = User(username='sowkya', email='sowkya@example.com', password=hashed_password, role='admin')
        db.session.add(admin)
        db.session.commit()

# --- Routes ---

@app.route("/")
def index():
    if current_user.is_authenticated:
        # Provide available years for quick analytics access
        try:
            years_raw = db.session.query(func.strftime('%Y', Patient.date_posted)).distinct().all()
            years = sorted({int(y[0]) for y in years_raw if y[0]})
        except Exception:
            years = []

        # Selected year from query param (defaults to 2025 when not provided)
        selected_year = request.args.get('year')
        if selected_year and selected_year.isdigit():
            selected_year = int(selected_year)
        else:
            # Default KPI view target year as requested
            selected_year = datetime.now().year

        # Compute KPI metrics for selected_year if available
        kpis = {}
        labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        monthly_counts = [0]*12
        total_dr_cases = 0
        total_screened = 0
        growth_percent = None
        peak_month = None
        avg_monthly = 0

        if selected_year:
            try:
                # monthly counts
                for m in range(1,13):
                    cnt = Patient.query.filter(func.strftime('%Y', Patient.date_posted) == str(selected_year), func.strftime('%m', Patient.date_posted) == f"{m:02d}").count()
                    monthly_counts[m-1] = cnt

                # total DR cases (exclude 'No DR')
                total_dr_cases = Patient.query.filter(func.strftime('%Y', Patient.date_posted) == str(selected_year), Patient.dr_stage != 'No DR').count()

                # total patients screened
                total_screened = Patient.query.filter(func.strftime('%Y', Patient.date_posted) == str(selected_year)).count()

                # growth percent vs previous year
                prev_year = selected_year - 1
                prev_total = Patient.query.filter(func.strftime('%Y', Patient.date_posted) == str(prev_year), Patient.dr_stage != 'No DR').count()
                if prev_total > 0:
                    growth_percent = round(((total_dr_cases - prev_total) / prev_total) * 100, 1)
                else:
                    growth_percent = None

                # peak month
                if any(monthly_counts):
                    peak_idx = int(max(range(12), key=lambda i: monthly_counts[i]))
                    peak_month = labels[peak_idx]

                # average monthly cases (use 12 months average)
                avg_monthly = round(sum(monthly_counts) / 12, 1)

                # --- Granular Stage Analysis for Clinical Dashboard ---
                stages = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
                stage_counts = {s: 0 for s in stages}
                
                # Get all records for this year to aggregate stages
                year_records = Patient.query.filter(func.strftime('%Y', Patient.date_posted) == str(selected_year)).all()
                for rec in year_records:
                    if rec.dr_stage in stage_counts:
                        stage_counts[rec.dr_stage] += 1
                        
                # Specialized Metrics
                high_risk_cases = stage_counts['Severe'] + stage_counts['Proliferative DR']
                healthy_cases = stage_counts['No DR']

            except Exception:
                stage_counts = {s: 0 for s in ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']}
                high_risk_cases = 0
                healthy_cases = 0
                pass

        kpis = {
            'total_dr_cases': total_dr_cases,
            'growth_percent': growth_percent,
            'peak_month': peak_month,
            'avg_monthly': avg_monthly,
            'total_screened': total_screened,
            'selected_year': selected_year,
            'labels': labels,
            'monthly_counts': monthly_counts,
            'high_risk_cases': high_risk_cases,
            'healthy_cases': healthy_cases,
            'stage_counts': stage_counts
        }

        # Also try to get years from external CSV dataset if present (for external analytics card)
        csv_path = os.path.join(app.root_path, 'data', 'sample_dr_cases.csv')
        external_years = []
        if os.path.exists(csv_path):
            external_years = list(range(2010, 2025))
        else:
            external_years = []
            
        # Select recent 5 patients for the "Recent Activity" table
        recent_patients = Patient.query.order_by(Patient.date_posted.desc()).limit(5).all()
            
        return render_template('index.html', title='Home', years=years, external_years=external_years, kpis=kpis, recent_patients=recent_patients)
    
    # If not authenticated, just render index with empty data (template will handle this)
    # If not authenticated, provide default empty KPIs structure to prevent template errors
    default_kpis = {
        'total_dr_cases': 0, 'growth_percent': None, 'peak_month': None, 'avg_monthly': 0, 
        'total_screened': 0, 'selected_year': None, 'labels': [], 'monthly_counts': [], 
        'high_risk_cases': 0, 'healthy_cases': 0, 
        'stage_counts': {'No DR':0, 'Mild':0, 'Moderate':0, 'Severe':0, 'Proliferative DR':0}
    }
    return render_template('index.html', title='Home', years=[], external_years=[], kpis=default_kpis, recent_patients=[])

@app.route("/information")
def information():
    return render_template('information.html', title='Information')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            if user.is_blocked:
                flash('Your account has been blocked. Please contact admin.', 'danger')
                return redirect(url_for('login'))
                
            login_user(user)
            flash('Login Successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
            
    return render_template('login.html', title='Login')
    return render_template('login.html', title='Login')

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash('Username or Email already exists. Please choose a different one.', 'danger')
        else:
            hashed_password = generate_password_hash(password)
            user = User(username=username, email=email, password=hashed_password, role='user')
            db.session.add(user)
            db.session.commit()
            flash('Account created! You can now login.', 'success')
            return redirect(url_for('login'))
            
    return render_template('register.html', title='Register')
@app.route("/dashboard")
@login_required
def dashboard():
    # Provide available years for quick analytics access
    years_raw = db.session.query(func.strftime('%Y', Patient.date_posted)).distinct().all()
    years = sorted({int(y[0]) for y in years_raw if y[0]})

    # Selected year from query param (defaults to 2025 when not provided)
    selected_year = request.args.get('year')
    if selected_year and selected_year.isdigit():
        selected_year = int(selected_year)
    else:
        # Default KPI view target year as requested
        selected_year = 2025

    # Compute KPI metrics for selected_year if available
    kpis = {}
    labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthly_counts = [0]*12
    total_dr_cases = 0
    total_screened = 0
    growth_percent = None
    peak_month = None
    avg_monthly = 0

    if selected_year:
        # monthly counts
        for m in range(1,13):
            cnt = Patient.query.filter(func.strftime('%Y', Patient.date_posted) == str(selected_year), func.strftime('%m', Patient.date_posted) == f"{m:02d}").count()
            monthly_counts[m-1] = cnt

        # total DR cases (exclude 'No DR')
        total_dr_cases = Patient.query.filter(func.strftime('%Y', Patient.date_posted) == str(selected_year), Patient.dr_stage != 'No DR').count()

        # total patients screened
        total_screened = Patient.query.filter(func.strftime('%Y', Patient.date_posted) == str(selected_year)).count()

        # growth percent vs previous year
        prev_year = selected_year - 1
        prev_total = Patient.query.filter(func.strftime('%Y', Patient.date_posted) == str(prev_year), Patient.dr_stage != 'No DR').count()
        if prev_total > 0:
            growth_percent = round(((total_dr_cases - prev_total) / prev_total) * 100, 1)
        else:
            growth_percent = None

        # peak month
        if any(monthly_counts):
            peak_idx = int(max(range(12), key=lambda i: monthly_counts[i]))
            peak_month = labels[peak_idx]

        # average monthly cases (use 12 months average)
        avg_monthly = round(sum(monthly_counts) / 12, 1)

        # --- Granular Stage Analysis for Clinical Dashboard ---
        stages = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        stage_counts = {s: 0 for s in stages}
                
        try:
            # Get all records for this year to aggregate stages
            year_records = Patient.query.filter(func.strftime('%Y', Patient.date_posted) == str(selected_year)).all()
            for rec in year_records:
                if rec.dr_stage in stage_counts:
                    stage_counts[rec.dr_stage] += 1
        except Exception:
            pass
                        
        # Specialized Metrics
        high_risk_cases = stage_counts['Severe'] + stage_counts['Proliferative DR']
        healthy_cases = stage_counts['No DR']

    kpis = {
        'total_dr_cases': total_dr_cases,
        'growth_percent': growth_percent,
        'peak_month': peak_month,
        'avg_monthly': avg_monthly,
        'total_screened': total_screened,
        'selected_year': selected_year,
        'labels': labels,
        'monthly_counts': monthly_counts,
        'high_risk_cases': high_risk_cases,
        'healthy_cases': healthy_cases,
        'stage_counts': stage_counts
    }

    # Also try to get years from external CSV dataset if present (for external analytics card)
    csv_path = os.path.join(app.root_path, 'data', 'sample_dr_cases.csv')
    external_years = []
    if os.path.exists(csv_path):
        # Present a fixed range for external analytics selection (2010-2024)
        external_years = list(range(2010, 2025))
    else:
        external_years = []

    # Select recent 5 patients for the "Recent Activity" table
    recent_patients = Patient.query.order_by(Patient.date_posted.desc()).limit(5).all()

    return render_template('index.html', title='Home', years=years, external_years=external_years, kpis=kpis, recent_patients=recent_patients)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route("/predict", methods=['GET', 'POST'])
@login_required
def predict():
    # Step 1: Patient Basic Info
    return render_template('patient_form.html', title='Patient Details')

@app.route("/health_check", methods=['GET', 'POST'])
@login_required
def health_check():
    # Step 2: Health Params & Image Upload
    if request.method == 'POST':
        # Retrieve data from Step 1
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        email = request.form.get('email')
        phone = request.form.get('phone')
        blood_group = request.form.get('blood_group')
        address = request.form.get('address')

        return render_template('health_form.html', title='Health Parameters', 
                       name=name, age=age, gender=gender,
                       email=email, phone=phone, blood_group=blood_group,
                       address=address)

    # If not POST, redirect back to the patient details form
    return redirect(url_for('predict'))

@app.route("/result", methods=['POST'])
@login_required
def result():
    if request.method == 'POST':
        # Get all data
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        email = request.form.get('email')
        phone = request.form.get('phone')
        blood_group = request.form.get('blood_group')
        address = request.form.get('address')
        
        # Parse numeric inputs with validation
        parse_errors = []
        try:
            systolic = int(request.form.get('systolic'))
        except Exception:
            parse_errors.append('Systolic BP must be an integer')
            systolic = None
        try:
            diastolic = int(request.form.get('diastolic'))
        except Exception:
            parse_errors.append('Diastolic BP must be an integer')
            diastolic = None
        try:
            hba1c = float(request.form.get('hba1c'))
        except Exception:
            parse_errors.append('HbA1c must be a number (e.g. 6.5)')
            hba1c = None
        try:
            cholesterol = int(request.form.get('cholesterol'))
        except Exception:
            parse_errors.append('Cholesterol must be an integer')
            cholesterol = None

        # Age validation
        try:
            age_val = int(age) if age is not None else None
        except Exception:
            age_val = None
            parse_errors.append('Age must be an integer')

        # If parsing errors, show them
        if parse_errors:
            return render_template('result.html', title='Validation Error', error='Data Validation Failed', image_file=None, patient=None, reasons=parse_errors)

        # Range checks (as requested)
        validation_errors = []
        if age_val is None or age_val < 1 or age_val > 120:
            validation_errors.append('Age must be between 1 and 120')
        if systolic is None or systolic < 80 or systolic > 200:
            validation_errors.append('Systolic BP must be between 80 and 200 mmHg')
        if diastolic is None or diastolic < 50 or diastolic > 120:
            validation_errors.append('Diastolic BP must be between 50 and 120 mmHg')
        if hba1c is None or hba1c < 3 or hba1c > 15:
            validation_errors.append('HbA1c must be between 3.0 and 15.0 (%)')
        if cholesterol is None or cholesterol < 80 or cholesterol > 400:
            validation_errors.append('Cholesterol must be between 80 and 400 mg/dL')

        if validation_errors:
            return render_template('result.html', title='Validation Error', error='Data Validation Failed', image_file=None, patient=None, reasons=validation_errors)
        
        # Image Upload
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Validate Image
            from utils import validate_image
            if not validate_image(file_path):
                # Invalid Image Case
                return render_template('result.html', title='Prediction Error',
                                       error="Invalid Image Type",
                                       reasons=["Uploaded file is not a valid image.", "Please upload a retinal fundus photo."],
                                       image_file=None,
                                       patient=None)

            # Logic
            dr_stage, confidence, heatmap_filename = predict_dr_stage(file_path)
            suitability, reasons = check_operation_suitability(systolic, diastolic, hba1c, cholesterol, dr_stage)
            
            # Save to DB
            patient = Patient(name=name, age=age, gender=gender,
                              phone=phone, email=email, blood_group=blood_group,
                              systolic_bp=systolic, diastolic_bp=diastolic,
                              hba1c=hba1c, cholesterol=cholesterol,
                              image_file=filename, dr_stage=dr_stage,
                              confidence=confidence,
                              operation_status=suitability,
                              user_id=current_user.id)
            db.session.add(patient)
            db.session.commit()
            
            # Get detailed clinical findings for specific feedback
            findings = get_clinical_findings(dr_stage)
            
            return render_template('result.html', title='Prediction Result', 
                                   patient=patient, reasons=reasons, 
                                   image_file=filename, heatmap_file=heatmap_filename,
                                   findings=findings)
            
    return redirect(url_for('dashboard'))

@app.route("/history")
@login_required
def history():
    # Support searching by Patient UID or name via query param 'q'
    q = request.args.get('q', '').strip()
    # Filter by Severity
    severity = request.args.get('severity', 'All')
    
    query = Patient.query
    
    if q:
        if q.isdigit():
             query = query.filter(Patient.id == int(q))
        else:
             query = query.filter(or_(Patient.patient_uid.ilike(f"%{q}%"), Patient.name.ilike(f"%{q}%")))
             
    if severity and severity != 'All':
        query = query.filter(Patient.dr_stage == severity)
        
    patients = query.order_by(Patient.date_posted.desc()).all()

    return render_template('history.html', title='Patient History', patients=patients, q=q, severity=severity)


@app.route('/analytics')
@login_required
def analytics():
    # Year selection
    years_raw = db.session.query(func.strftime('%Y', Patient.date_posted)).distinct().all()
    years = sorted({int(y[0]) for y in years_raw if y[0]})
    if not years:
        # No data
        return render_template('analytics.html', title='Analytics', years=[], selected_year=None, labels=[], actual=[], predicted=[])

    selected_year = request.args.get('year')
    if selected_year and selected_year.isdigit():
        selected_year = int(selected_year)
    else:
        selected_year = years[-1]

    # Prepare actual counts per month for selected year
    labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    actual = [0]*12
    patients_in_year = Patient.query.filter(func.strftime('%Y', Patient.date_posted) == str(selected_year)).all()
    for p in patients_in_year:
        if p.date_posted and isinstance(p.date_posted, type(p.date_posted)):
            m = p.date_posted.month
            if 1 <= m <= 12:
                actual[m-1] += 1

    # Prediction per month using simple linear regression (years vs counts for that month)
    predicted = [0]*12
    all_years = years
    for month_idx in range(1,13):
        # Build X (years) and Y (counts) for this month
        X = []
        Y = []
        for y in all_years:
            cnt = Patient.query.filter(func.strftime('%Y', Patient.date_posted) == str(y), func.strftime('%m', Patient.date_posted) == f"{month_idx:02d}").count()
            X.append(y)
            Y.append(cnt)

        if len(X) >= 2 and any(Y):
            try:
                coef = np.polyfit(X, Y, 1)
                slope, intercept = coef[0], coef[1]
                pred = int(max(0, round(slope * selected_year + intercept)))
            except Exception:
                pred = int(round(sum(Y)/len(Y))) if Y else 0
        else:
            pred = int(round(sum(Y)/len(Y))) if Y else 0

        predicted[month_idx-1] = pred

    return render_template('analytics.html', title='Analytics', years=years, selected_year=selected_year, labels=labels, actual=actual, predicted=predicted)


@app.route('/external-analytics')
@login_required
def external_analytics():
    # CSV file provided in /data/sample_dr_cases.csv (sample external dataset)
    csv_path = os.path.join(app.root_path, 'data', 'sample_dr_cases.csv')
    # If user uploaded dataset exists, prefer it
    user_csv = os.path.join(app.root_path, 'data', 'external_dataset.csv')
    if os.path.exists(user_csv):
        csv_path = user_csv
    if not os.path.exists(csv_path):
        return render_template('external_analytics.html', title='External Analytics', years=[], selected_year=None, labels=[], actual=[], predicted=[], message='Dataset not found')

    # Parse CSV and aggregate by year-month
    data = []
    years_set = set()
    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                d = datetime.strptime(r['date'], '%Y-%m-%d')
                cnt = int(r['count'])
                data.append((d, cnt))
                years_set.add(d.year)
            except Exception:
                continue

    # Use a fixed year range for external analytics (2010-2024)
    # The range(2010, 2025) produces 2010 up to 2024 inclusive.
    years = list(range(2010, 2025))
    if not years:
        return render_template('external_analytics.html', title='External Analytics', years=[], selected_year=None, labels=[], actual=[], predicted=[], message='No data in dataset')

    selected_year = request.args.get('year')
    if selected_year and selected_year.isdigit():
        selected_year = int(selected_year)
    else:
        selected_year = years[-1]

    labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    # actual counts for selected year
    actual = [0]*12
    for d,cnt in data:
        if d.year == selected_year:
            actual[d.month-1] += cnt

    # prediction across years per month using simple linear regression
    predicted = [0]*12
    for month_idx in range(1,13):
        X = []
        Y = []
        for y in years:
            s = sum(cnt for d,cnt in data if d.year==y and d.month==month_idx)
            X.append(y)
            Y.append(s)
        if len(X) >= 2 and any(Y):
            try:
                coef = np.polyfit(X, Y, 1)
                slope, intercept = coef[0], coef[1]
                pred = int(max(0, round(slope * selected_year + intercept)))
            except Exception:
                pred = int(round(sum(Y)/len(Y))) if Y else 0
        else:
            pred = int(round(sum(Y)/len(Y))) if Y else 0
        
        predicted[month_idx-1] = pred

    return render_template('external_analytics.html', title='External Analytics', years=years, selected_year=selected_year, labels=labels, actual=actual, predicted=predicted, message=None)


@app.route('/upload_external', methods=['GET', 'POST'])
@login_required
def upload_external():
    if current_user.role != 'admin':
        flash('Access denied. Admins only.', 'danger')
        return redirect(url_for('index'))
        
    # Dedicated upload page: on GET show the form, on POST accept CSV upload and save as data/external_dataset.csv
    if request.method == 'POST':
        file = request.files.get('dataset')
        if not file or file.filename == '':
            flash('No file selected for upload.', 'warning')
            return redirect(url_for('upload_external'))

        filename = secure_filename(file.filename)
        if not filename.lower().endswith('.csv'):
            flash('Only CSV files are supported.', 'danger')
            return redirect(url_for('upload_external'))

        data_dir = os.path.join(app.root_path, 'data')
        os.makedirs(data_dir, exist_ok=True)
        save_path = os.path.join(data_dir, 'external_dataset.csv')
        try:
            file.save(save_path)
            flash('External dataset uploaded successfully.', 'success')
        except Exception as e:
            flash(f'Failed to save uploaded file: {e}', 'danger')

        return redirect(url_for('external_analytics'))

    return render_template('upload_external.html', title='Upload External Dataset')


@app.route('/patient/<int:patient_id>')
@login_required
def view_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    findings = get_clinical_findings(patient.dr_stage)
    return render_template('view_patient.html', patient=patient, title='Patient Details', is_admin=(current_user.role == 'admin'), findings=findings)




@app.route('/patient/<int:patient_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_patient(patient_id):
    if current_user.role != 'admin':
        flash('Access denied. Admins only.', 'danger')
        return redirect(url_for('view_patient', patient_id=patient_id))
        
    patient = Patient.query.get_or_404(patient_id)
    if request.method == 'POST':
        # Update editable fields with validation
        patient.name = request.form.get('name') or patient.name
        age = request.form.get('age')
        # Collect validation errors and only commit if none
        errors = []

        if age:
            try:
                age_int = int(age)
                if age_int < 1 or age_int > 120:
                    errors.append('Age must be between 1 and 120')
                else:
                    patient.age = age_int
            except Exception:
                errors.append('Age must be an integer')

        patient.gender = request.form.get('gender') or patient.gender
        patient.email = request.form.get('email') or patient.email
        patient.phone = request.form.get('phone') or patient.phone
        patient.address = request.form.get('address') or patient.address
        patient.blood_group = request.form.get('blood_group') or patient.blood_group

        # clinical updates with validation
        syst = request.form.get('systolic')
        dias = request.form.get('diastolic')
        h = request.form.get('hba1c')
        chol = request.form.get('cholesterol')

        if syst:
            try:
                s_val = int(syst)
                if s_val < 80 or s_val > 200:
                    errors.append('Systolic BP must be between 80 and 200 mmHg')
                else:
                    patient.systolic_bp = s_val
            except Exception:
                errors.append('Systolic BP must be an integer')

        if dias:
            try:
                d_val = int(dias)
                if d_val < 50 or d_val > 120:
                    errors.append('Diastolic BP must be between 50 and 120 mmHg')
                else:
                    patient.diastolic_bp = d_val
            except Exception:
                errors.append('Diastolic BP must be an integer')

        if h:
            try:
                h_val = float(h)
                if h_val < 3 or h_val > 15:
                    errors.append('HbA1c must be between 3.0 and 15.0 (%)')
                else:
                    patient.hba1c = h_val
            except Exception:
                errors.append('HbA1c must be a number')

        if chol:
            try:
                c_val = int(chol)
                if c_val < 80 or c_val > 400:
                    errors.append('Cholesterol must be between 80 and 400 mg/dL')
                else:
                    patient.cholesterol = c_val
            except Exception:
                errors.append('Cholesterol must be an integer')

        if errors:
            for e in errors:
                flash(e, 'danger')
            # Render the edit form again showing attempted values (not committed)
            return render_template('edit_patient.html', patient=patient, title='Edit Patient')

        db.session.commit()
        flash('Patient record updated.', 'success')
        return redirect(url_for('history'))

    return render_template('edit_patient.html', patient=patient, title='Edit Patient')


@app.route('/patient/<int:patient_id>/delete', methods=['POST'])
@login_required
def delete_patient(patient_id):
    if current_user.role != 'admin':
        flash('Access denied. Admins only.', 'danger')
        return redirect(url_for('view_patient', patient_id=patient_id))
        
    patient = Patient.query.get_or_404(patient_id)
    # Try remove uploaded image file (if stored under UPLOAD_FOLDER)
    try:
        if patient.image_file:
            img_path = os.path.join(app.config.get('UPLOAD_FOLDER', ''), patient.image_file)
            if os.path.exists(img_path):
                os.remove(img_path)
    except Exception:
        # ignore file removal errors
        pass

    db.session.delete(patient)
    db.session.commit()
    flash('Patient record deleted.', 'success')
    return redirect(url_for('history'))

@app.route('/admin/users')
@login_required
def admin_users():
    if current_user.role != 'admin':
        flash('Access denied. Admins only.', 'danger')
        return redirect(url_for('index'))
    
    users = User.query.all()
    # Create a list of tuples (user, prediction_count)
    user_data = []
    for user in users:
        count = Patient.query.filter_by(user_id=user.id).count()
        user_data.append((user, count))
        
    return render_template('admin_users.html', title='Manage Users', users=user_data)

@app.route('/admin/user/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
def admin_edit_user(user_id):
    if current_user.role != 'admin':
        flash('Access denied. Admins only.', 'danger')
        return redirect(url_for('index'))
        
    user = User.query.get_or_404(user_id)
    if request.method == 'POST':
        # Update Block Status
        is_blocked = request.form.get('is_blocked') == 'on'
        user.is_blocked = is_blocked
        
        # Update Password if provided
        new_password = request.form.get('password')
        if new_password:
             user.password = generate_password_hash(new_password)
             flash(f'Password updated for user {user.username}', 'success')
             
        db.session.commit()
        flash(f'User {user.username} updated.', 'success')
        return redirect(url_for('admin_users'))
        
    return render_template('admin_edit_user.html', title='Edit User', user=user)

@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
def admin_delete_user(user_id):
    if current_user.role != 'admin':
        flash('Access denied. Admins only.', 'danger')
        return redirect(url_for('index'))
        
    user = User.query.get_or_404(user_id)
    if user.role == 'admin':
        flash('Cannot delete admin account.', 'danger')
        return redirect(url_for('admin_users'))
        
    # Optional: Delete associated patients or handle orphans?
    # For now, we leave patients or maybe set user_id to None if we wanted.
    # But usually deleting user logic depends on requirements. 
    # Let's keep patients but unlink them to safe data, or delete them.
    # User asked: "delete user". Safe approach: delete user, keep records unlinked or cascade.
    # Given requirements, simple delete is fine.
    
    db.session.delete(user)
    db.session.commit()
    flash('User deleted.', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/user/<int:user_id>/activity')
@login_required
def admin_user_activity(user_id):
    if current_user.role != 'admin':
        flash('Access denied. Admins only.', 'danger')
        return redirect(url_for('index'))
    
    user = User.query.get_or_404(user_id)
    # Fetch all patients predicted by this user, ordered by date desc
    patients = Patient.query.filter_by(user_id=user.id).order_by(Patient.date_posted.desc()).all()
    
    return render_template('admin_user_details.html', user=user, patients=patients, title=f"Activity Log - {user.username}")

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)