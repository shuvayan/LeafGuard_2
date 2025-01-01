import google.generativeai as genai

apk = 'AIzaSyD5z3d_Fwl4YUWB8qe-cQGGUVeuSriSFfc'

class LLMService:
    def __init__(self, api_key):
        genai.configure(api_key=apk)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def get_disease_treatment(self, disease_name):
        try:
            prompt = f"""Generate a detailed plant disease analysis report for: {disease_name}
            [Your existing prompt template...]"""
            
            response = self.model.generate_content(prompt)
            return response.text if response.text else "Unable to generate treatment information."
        except Exception as e:
            return f"Treatment information unavailable: {str(e)}"