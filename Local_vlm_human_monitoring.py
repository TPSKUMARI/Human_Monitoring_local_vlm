import cv2
from PIL import Image
import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import time
import threading
from queue import Queue
import sqlite3
from datetime import datetime
import os

class VideoRecorder:
    def __init__(self, output_folder="output"):
        self.output_folder = output_folder
        self.recording = False
        self.video_writer = None
        self.current_filename = None
        self.record_start_time = None
        self.last_recording_time = 0
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created output folder: {self.output_folder}")
    
    def should_start_recording(self):
        """Check if it's time to start a new recording (every 4 minutes)"""
        current_time = time.time()
        return (current_time - self.last_recording_time) >= 240  # 240 seconds = 4 minutes
    
    def start_recording(self, frame_width, frame_height, fps=30):
        """Start recording a new video"""
        if self.recording:
            return  # Already recording
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_filename = os.path.join(self.output_folder, f"monitoring_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.current_filename, fourcc, fps, (frame_width, frame_height))
        
        if self.video_writer.isOpened():
            self.recording = True
            self.record_start_time = time.time()
            self.last_recording_time = time.time()
            print(f"[RECORDING] Started: {self.current_filename}")
        else:
            print(f"[ERROR] Failed to start recording: {self.current_filename}")
    
    def write_frame(self, frame):
        """Write a frame to the current video"""
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)
    
    def should_stop_recording(self):
        """Check if we should stop recording (after 1 minute)"""
        if not self.recording:
            return False
        return (time.time() - self.record_start_time) >= 60  # 60 seconds = 1 minute
    
    def stop_recording(self):
        """Stop current recording"""
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
            self.recording = False
            duration = time.time() - self.record_start_time
            print(f"[RECORDING] Finished: {self.current_filename} (Duration: {duration:.1f}s)")
            self.video_writer = None
            self.current_filename = None
    
    def cleanup(self):
        """Clean up resources"""
        if self.recording:
            self.stop_recording()

class SimpleDatabase:
    def __init__(self, db_path="person_monitoring.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize simple database"""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            print("Removed old database")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE monitoring_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                datetime_full TEXT NOT NULL,
                activity_type TEXT NOT NULL,
                safety_level TEXT NOT NULL,
                analysis_result TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"Simple database created: {self.db_path}")
    
    def save_analysis(self, activity_type, safety_level, analysis_result):
        """Save analysis to simple database"""
        try:
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time_only = now.strftime("%H:%M:%S")
            datetime_full = now.strftime("%Y-%m-%d %H:%M:%S")

            cleaned_analysis = analysis_result.replace("'", "''").replace('"', '""')[:1000]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO monitoring_log (date, time, datetime_full, activity_type, safety_level, analysis_result)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (date, time_only, datetime_full, activity_type, safety_level, cleaned_analysis))
            
            conn.commit()
            conn.close()
            
            print(f"[DB] Saved: {datetime_full} - {activity_type} - {safety_level}")
            
        except Exception as e:
            print(f"[DB ERROR] Failed to save: {str(e)}")
    
    def get_recent_activities(self, hours=24):
        """Get recent activities"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT datetime_full, activity_type, safety_level, analysis_result 
            FROM monitoring_log 
            WHERE datetime(datetime_full) > datetime('now', '-{} hours')
            ORDER BY datetime_full DESC
        '''.format(hours))
        
        results = cursor.fetchall()
        conn.close()
        return results

class EnhancedContinuousVLM:
    def __init__(self):
        self.model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.db = SimpleDatabase()
        self.video_recorder = VideoRecorder()

        self.analysis_interval = 8
        self.frame_buffer = Queue(maxsize=12)
        self.latest_analysis = "Initializing comprehensive monitoring system..."
        self.latest_activity = "No activity detected"
        self.latest_safety = "Unknown"
        self.running = False
        
        self.load_model()
    
    def load_model(self):
        print("Loading Enhanced VLM model...")
        print(f"Using device: {self.device}")
        
        self.processor = LlavaProcessor.from_pretrained(self.model_id)
        
        if self.device == "cuda":
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
        
        self.model.eval()
        print("Enhanced model loaded successfully!")
    
    def capture_frames_continuously(self, cap):
        """Continuously capture frames in background thread"""
        while self.running:
            ret, frame = cap.read()
            if ret:
                small_frame = cv2.resize(frame, (384, 288))
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                if self.frame_buffer.full():
                    try:
                        self.frame_buffer.get_nowait()
                    except:
                        pass
                
                try:
                    self.frame_buffer.put_nowait(pil_image)
                except:
                    pass
            
            time.sleep(0.08)
    
    def analyze_frames_periodically(self):
        """Analyze frames every few seconds in background thread"""
        while self.running:
            time.sleep(self.analysis_interval)
            
            if not self.frame_buffer.empty():
                frames = []
                try:
                    for _ in range(min(3, self.frame_buffer.qsize())):
                        frames.append(self.frame_buffer.get_nowait())
                except:
                    pass
                
                if frames:
                    print(f"[ANALYSIS] Processing {len(frames)} frames...")
                    try:
                        analysis = self.analyze_frames(frames)

                        activity_type = self.extract_activity_type(analysis)
                        safety_level = self.extract_safety_level(analysis)

                        self.latest_analysis = analysis
                        self.latest_activity = activity_type
                        self.latest_safety = safety_level

                        self.db.save_analysis(activity_type, safety_level, analysis)

                        print(f"[ANALYSIS] Activity: {activity_type} | Safety: {safety_level}")

                    except Exception as e:
                        print(f"[ERROR] Analysis failed: {str(e)}")
                        continue
    
    def analyze_frames(self, frames):
        """Enhanced analysis with comprehensive prompt"""
        try:
            user_prompt = """
            Perform a comprehensive safety and wellness assessment of these sequential images. Provide detailed analysis for each of the following areas:

            1. PERSON DETECTION & COUNT: Identify all individuals present, their approximate age, gender, and positioning in the space.

            2. DETAILED ACTIVITY ANALYSIS: Describe the specific activity being performed (sitting, standing, walking, lying, eating, reading, watching TV, using devices, exercising, cleaning, cooking, etc.). Include body language, posture, and movement patterns.

            3. COMPREHENSIVE SAFETY ASSESSMENT: Evaluate immediate safety risks including fall hazards, environmental dangers, medical emergencies, distress signs, unusual behaviors, or emergency situations. Look for signs of confusion, disorientation, or distress.

            4. HEALTH & WELLNESS INDICATORS: Assess visible health status including alertness level, mobility, coordination, facial expressions, breathing patterns, and overall physical condition. Note any signs of pain, fatigue, or illness.

            5. ENVIRONMENTAL SAFETY EVALUATION: Examine the surroundings for hazards such as obstacles, poor lighting, slippery surfaces, accessibility issues, medication visibility, or dangerous items within reach.

            6. BEHAVIORAL & EMOTIONAL STATE: Analyze facial expressions, body language, and behavioral patterns. Assess mood, stress levels, social interaction, and cognitive engagement.

            7. CARE RECOMMENDATIONS: Provide specific suggestions for immediate attention, safety improvements, or wellness interventions based on your observations.

            Provide a thorough, detailed response covering all these aspects with specific observations and actionable insights.
            """
            
            prompt = f"<|im_start|>user{'<image>' * len(frames)}\n{user_prompt}<|im_end|><|im_start|>assistant"
            
            inputs = self.processor(text=prompt, images=frames, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if hasattr(self.model, 'dtype'):
                    inputs = {k: v.to(self.model.dtype) if v.dtype.is_floating_point else v 
                            for k, v in inputs.items()}
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.3,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
            
            response = self.processor.decode(output[0], skip_special_tokens=True)
            
            if "assistant" in response:
                result = response.split("assistant")[-1].strip()
            else:
                result = response.strip()
            
            return result
            
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return error_msg
    
    def extract_activity_type(self, analysis):
        """Extract activity type from analysis with better keyword matching"""
        analysis_lower = analysis.lower()

        activities = {
            'sitting_reading': ['sitting', 'seated', 'reading', 'book', 'literature', 'text'],
            'sitting_watching': ['sitting', 'seated', 'watching', 'tv', 'television', 'screen', 'viewing'],
            'sitting_resting': ['sitting', 'seated', 'resting', 'chair', 'relaxing', 'positioned'],
            'standing': ['standing', 'upright', 'vertical', 'erect'],
            'walking': ['walking', 'moving', 'motion', 'movement', 'stepping'],
            'lying_down': ['lying', 'laying', 'bed', 'sleeping', 'horizontal', 'supine'],
            'eating': ['eating', 'drinking', 'food', 'meal', 'consuming', 'dining'],
            'cooking': ['cooking', 'kitchen', 'preparing', 'culinary'],
            'cleaning': ['cleaning', 'tidying', 'organizing', 'housework'],
            'exercising': ['exercise', 'stretching', 'workout', 'physical activity'],
            'using_phone': ['phone', 'calling', 'talking', 'mobile', 'device'],
            'using_computer': ['computer', 'laptop', 'device', 'technology'],
            'personal_care': ['washing', 'grooming', 'bathroom', 'hygiene'],
            'no_person': ['no person', 'empty', 'nobody', 'vacant', 'unoccupied']
        }
        
        for activity, keywords in activities.items():
            matches = sum(1 for kw in keywords if kw in analysis_lower)
            if matches >= 1:
                return activity

        return 'general_activity'
    
    def extract_safety_level(self, analysis):
        """Extract safety level from analysis"""
        analysis_lower = analysis.lower()
        
        high_risk = ['fallen', 'fall', 'emergency', 'distress', 'help', 'danger', 'unconscious']
        medium_risk = ['unsteady', 'confusion', 'disoriented', 'pain', 'difficulty']
        safe = ['stable', 'alert', 'normal', 'safe', 'good', 'healthy']
        
        if any(word in analysis_lower for word in high_risk):
            return 'high_risk'
        elif any(word in analysis_lower for word in medium_risk):
            return 'medium_risk'
        elif any(word in analysis_lower for word in safe):
            return 'safe'
        else:
            return 'normal'
    
    def draw_enhanced_overlay(self, frame):
        """Draw enhanced overlay with better colors and layout"""
        height, width = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "Person Monitoring System", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Time: {timestamp}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        activity_color = (0, 255, 0) if self.latest_safety == "safe" else (0, 165, 255) if self.latest_safety == "normal" else (0, 0, 255)
        cv2.putText(frame, f"Activity: {self.latest_activity.replace('_', ' ').title()}", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, activity_color, 2)
        
        safety_color = (0, 255, 0) if self.latest_safety == "safe" else (0, 165, 255) if self.latest_safety == "normal" else (0, 0, 255)
        cv2.putText(frame, f"Safety: {self.latest_safety.upper()}", (300, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, safety_color, 2)
        
        if self.video_recorder.recording:
            cv2.circle(frame, (width - 30, 30), 15, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (width - 70, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if self.video_recorder.record_start_time:
                elapsed = time.time() - self.video_recorder.record_start_time
                cv2.putText(frame, f"Recording: {elapsed:.0f}s/60s", (width - 200, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            if hasattr(self.video_recorder, 'last_recording_time'):
                time_until_next = 240 - (time.time() - self.video_recorder.last_recording_time)
                if time_until_next > 0:
                    cv2.putText(frame, f"Next recording: {time_until_next:.0f}s", (width - 200, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else:
                    cv2.putText(frame, "Recording starting...", (width - 200, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        analysis_lines = []
        if len(self.latest_analysis) > 150:
            words = self.latest_analysis.split()
            line = ""
            for word in words:
                if len(line + word) < 85:
                    line += word + " "
                else:
                    analysis_lines.append(line.strip())
                    line = word + " "
                    if len(analysis_lines) >= 3:
                        break
            if line and len(analysis_lines) < 3:
                analysis_lines.append(line.strip())
        else:
            analysis_lines = [self.latest_analysis[i:i+85] for i in range(0, len(self.latest_analysis), 85)][:3]
        
        y_start = 100
        for i, line in enumerate(analysis_lines):
            y_pos = y_start + i*15
            if y_pos < height - 30:
                cv2.putText(frame, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        
        return frame
    
    def run_continuous_monitoring(self):
        """Enhanced continuous webcam monitoring with database logging and video recording"""
        print("Starting Enhanced Person Monitoring System...")
        print("Controls: 'q' to quit, 's' for detailed analysis, 'r' for recent activities, 'v' to manually start recording")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set webcam properties
        frame_width = 1280
        frame_height = 720
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        
        capture_thread = threading.Thread(target=self.capture_frames_continuously, args=(cap,))
        analysis_thread = threading.Thread(target=self.analyze_frames_periodically)
        
        capture_thread.daemon = True
        analysis_thread.daemon = True
        
        capture_thread.start()
        analysis_thread.start()
        
        print(f"Enhanced monitoring started! Analysis every {self.analysis_interval} seconds...")
        print(f"Database: {self.db.db_path}")
        print(f"Video recording: 1-minute videos every 4 minutes to '{self.video_recorder.output_folder}' folder")
        
        try:
            while True:
                ret, frame = cap.read()
                if ret:
                    frame_with_overlay = self.draw_enhanced_overlay(frame)

                    if self.video_recorder.should_start_recording():
                        self.video_recorder.start_recording(frame_width, frame_height)

                    if self.video_recorder.recording:
                        self.video_recorder.write_frame(frame_with_overlay)

                        if self.video_recorder.should_stop_recording():
                            self.video_recorder.stop_recording()
                    
                    cv2.imshow('Person Monitoring System', frame_with_overlay)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    print(f"\n{'='*100}")
                    print(f"LATEST COMPREHENSIVE ANALYSIS")
                    print(f"{'='*100}")
                    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Activity: {self.latest_activity}")
                    print(f"Safety Level: {self.latest_safety}")
                    print(f"\nFull Analysis:\n{self.latest_analysis}")
                    print(f"{'='*100}\n")
                elif key == ord('r'):
                    self.print_recent_activities()
                elif key == ord('v'):
                    if not self.video_recorder.recording:
                        self.video_recorder.start_recording(frame_width, frame_height)
                        print("[MANUAL] Recording started")
                    else:
                        print("[MANUAL] Already recording...")
        
        except KeyboardInterrupt:
            pass
        
        finally:
            print("Stopping enhanced monitoring...")
            self.running = False

            self.video_recorder.cleanup()

            cap.release()
            cv2.destroyAllWindows()
            
            capture_thread.join(timeout=1)
            analysis_thread.join(timeout=1)
            
            print("Enhanced monitoring stopped.")
            print(f"Recorded videos saved in: {self.video_recorder.output_folder}")
    
    def print_recent_activities(self):
        """Print recent activities from database"""
        activities = self.db.get_recent_activities(24)
        print(f"\n{'='*80}")
        print(f"RECENT ACTIVITIES - Last 24 hours")
        print(f"{'='*80}")
        
        for i, (datetime_full, activity_type, safety_level, analysis_result) in enumerate(activities[:15]):
            print(f"\n[{i+1}] {datetime_full}")
            print(f"Activity: {activity_type.replace('_', ' ').title()}")
            print(f"Safety Level: {safety_level}")
            print(f"Analysis: {analysis_result[:150]}...")
            print("-" * 80)

def main():
    print("=== PERSON MONITORING SYSTEM WITH VIDEO RECORDING ===")
    print("Features: 300-token detailed analysis, simple database, automatic video recording")
    print("Database: Simple date/time/activity tracking")
    print("Analysis: 7-point detailed assessment framework")
    print("Video Recording: 1-minute videos every 4 minutes with full overlay details")
    print("Performance: GPU-optimized with enhanced UI")
    print("\nStarting comprehensive monitoring system...")
    
    vlm = EnhancedContinuousVLM()
    vlm.run_continuous_monitoring()

if __name__ == "__main__":
    main()