from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import uuid

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    role = db.Column(db.String(10), nullable=False, default='user')
    is_blocked = db.Column(db.Boolean, default=False, nullable=False) # New field to block users

    def __repr__(self):
        return f"User('{self.username}')"

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # Link prediction to a user (Nullable for backward compatibility)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    # Human-friendly unique identifier (UUID4) for easy lookup in UI
    patient_uid = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    
    # Health Parameters
    systolic_bp = db.Column(db.Integer, nullable=False)
    diastolic_bp = db.Column(db.Integer, nullable=False)
    hba1c = db.Column(db.Float, nullable=False)
    cholesterol = db.Column(db.Integer, nullable=False)
    
    # Files & Results
    image_file = db.Column(db.String(100), nullable=False, default='default.jpg')
    dr_stage = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=True) # Added confidence score
    operation_status = db.Column(db.String(100), nullable=False)
    
    # Contact & Bio
    phone = db.Column(db.String(15), nullable=False, default='N/A')
    email = db.Column(db.String(120), nullable=False, default='N/A')
    blood_group = db.Column(db.String(5), nullable=False, default='N/A')
    # Postal address for patient
    address = db.Column(db.String(200), nullable=False, default='N/A')
    
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"Patient('{self.name}', '{self.dr_stage}', '{self.date_posted}')"
