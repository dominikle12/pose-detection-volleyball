# Optimized Configuration for PyTorch OpenPose Ball Physics Demo
# This file centralizes all configuration parameters for easier tuning

import os

class Config:
    # Performance configuration
    KEYPOINT_CONFIDENCE_THRESHOLD = float(os.getenv('KEYPOINT_CONFIDENCE_THRESHOLD', '0.1'))
    POSE_SKIP_FRAMES = int(os.getenv('POSE_SKIP_FRAMES', '1'))
    RENDER_SKIP_FRAMES = int(os.getenv('RENDER_SKIP_FRAMES', '1'))
    USE_LOWER_RESOLUTION = os.getenv('USE_LOWER_RESOLUTION', 'True').lower() == 'true'
    DETECTION_SCALE_FACTOR = float(os.getenv('DETECTION_SCALE_FACTOR', '0.3'))
    
    # Physics configuration
    BALL_RADIUS = int(os.getenv('BALL_RADIUS', '20'))
    GRAVITY = float(os.getenv('GRAVITY', '0.9'))
    ELASTICITY = float(os.getenv('ELASTICITY', '0.7'))
    FRICTION = float(os.getenv('FRICTION', '0.98'))
    ARM_COLLISION_PADDING = int(os.getenv('ARM_COLLISION_PADDING', '10'))
    
    # Palm detection configuration
    PALM_BASE_SIZE = int(os.getenv('PALM_BASE_SIZE', '40'))
    MIN_PALM_SIZE = int(os.getenv('MIN_PALM_SIZE', '30'))
    MAX_PALM_SIZE = int(os.getenv('MAX_PALM_SIZE', '60'))
    PALM_DISTANCE_FACTOR = float(os.getenv('PALM_DISTANCE_FACTOR', '1.5'))
    WRIST_ELBOW_REF_DISTANCE = int(os.getenv('WRIST_ELBOW_REF_DISTANCE', '100'))
    
    # Debug and display flags
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    REDUCED_MODEL_PRECISION = os.getenv('REDUCED_MODEL_PRECISION', 'True').lower() == 'true'
    DISPLAY_KEYPOINTS = os.getenv('DISPLAY_KEYPOINTS', 'True').lower() == 'true'
    ENABLE_PALM_DETECTION = os.getenv('ENABLE_PALM_DETECTION', 'True').lower() == 'true'
    
    # Camera configuration
    CAMERA_FPS = int(os.getenv('CAMERA_FPS', '30'))
    CAMERA_BUFFER_SIZE = int(os.getenv('CAMERA_BUFFER_SIZE', '1'))
    CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '320'))  # Lower resolution for better performance
    CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '240'))
    
    # Physics optimization
    MAX_PHYSICS_STEPS = int(os.getenv('MAX_PHYSICS_STEPS', '3'))
    FIXED_TIME_STEP = float(os.getenv('FIXED_TIME_STEP', '0.016666'))  # 1/60 seconds
    
    # Collision optimization
    ARM_HISTORY_LENGTH = int(os.getenv('ARM_HISTORY_LENGTH', '3'))
    PALM_HISTORY_LENGTH = int(os.getenv('PALM_HISTORY_LENGTH', '5'))
    
    # Volleyball shot detection
    VOLLEYBALL_SHOT_COOLDOWN = float(os.getenv('VOLLEYBALL_SHOT_COOLDOWN', '1.0'))
    DIG_DISTANCE_THRESHOLD = int(os.getenv('DIG_DISTANCE_THRESHOLD', '150'))
    SET_HEIGHT_THRESHOLD = int(os.getenv('SET_HEIGHT_THRESHOLD', '100'))
    SPIKE_VELOCITY_THRESHOLD = int(os.getenv('SPIKE_VELOCITY_THRESHOLD', '15'))
    
    # Auto-reset configuration
    AUTO_RESET_DURATION = float(os.getenv('AUTO_RESET_DURATION', '3.0'))
    COUNTDOWN_DURATION = float(os.getenv('COUNTDOWN_DURATION', '3.0'))
    COUNTDOWN_SPAWN_HEIGHT = int(os.getenv('COUNTDOWN_SPAWN_HEIGHT', '50'))
    
    # Performance monitoring
    FPS_UPDATE_INTERVAL = float(os.getenv('FPS_UPDATE_INTERVAL', '1.0'))
    
    @classmethod
    def print_config(cls):
        """Print current configuration values"""
        print("=== Configuration ===")
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                print(f"{attr}: {getattr(cls, attr)}")
        print("==================")
    
    @classmethod
    def get_performance_profile(cls, profile_name='balanced'):
        """Get predefined performance profiles"""
        profiles = {
            'ultra_fast': {
                'POSE_SKIP_FRAMES': 12,
                'DETECTION_SCALE_FACTOR': 0.1,
                'DISPLAY_KEYPOINTS': False,
                'DEBUG_MODE': False,
                'ARM_HISTORY_LENGTH': 1,
                'PALM_HISTORY_LENGTH': 2,
                'KEYPOINT_CONFIDENCE_THRESHOLD': 0.2,
                'ENABLE_PALM_DETECTION': False,  # Disable palm detection for max speed
                'CAMERA_WIDTH': 320,
                'CAMERA_HEIGHT': 240,
            },
            'max_performance': {
                'POSE_SKIP_FRAMES': 8,
                'DETECTION_SCALE_FACTOR': 0.15,
                'DISPLAY_KEYPOINTS': False,
                'DEBUG_MODE': False,
                'ARM_HISTORY_LENGTH': 2,
                'PALM_HISTORY_LENGTH': 3,
                'KEYPOINT_CONFIDENCE_THRESHOLD': 0.15,
            },
            'balanced': {
                'POSE_SKIP_FRAMES': 3,
                'DETECTION_SCALE_FACTOR': 0.3,
                'DISPLAY_KEYPOINTS': True,
                'DEBUG_MODE': False,
                'ARM_HISTORY_LENGTH': 3,
                'PALM_HISTORY_LENGTH': 5,
            },
            'high_quality': {
                'POSE_SKIP_FRAMES': 1,
                'DETECTION_SCALE_FACTOR': 0.5,
                'DISPLAY_KEYPOINTS': True,
                'DEBUG_MODE': False,
                'ARM_HISTORY_LENGTH': 5,
                'PALM_HISTORY_LENGTH': 7,
            }
        }
        return profiles.get(profile_name, profiles['balanced'])