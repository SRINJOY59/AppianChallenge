import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

class EmailService:
    def __init__(self):
        load_dotenv()
        self.sender_email = os.getenv('EMAIL_ADDRESS')
        self.password = os.getenv('EMAIL_PASSWORD')
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587

    def send_email(self, recipient_email, subject, html_content):
        try:
            message = MIMEMultipart('alternative')
            message['Subject'] = subject
            message['From'] = self.sender_email
            message['To'] = recipient_email

            html_part = MIMEText(html_content, 'html')
            message.attach(html_part)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()  
                server.login(self.sender_email, self.password)
                
                server.send_message(message)
                return {"success": True, "message": "Email sent successfully"}

        except Exception as e:
            return {"success": False, "message": f"Failed to send email: {str(e)}"}

if __name__ == "__main__":

    html_template = """
    <!DOCTYPE html>
    <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
                <img src="https://your-logo-url.com/logo.png" alt="FinDoc.ai Logo" style="max-width: 150px; margin-bottom: 20px;">
                
                <h2 style="color: #2c3e50; margin-bottom: 20px;">Welcome to FinDoc.ai! 🎉</h2>
                
                <p style="color: #34495e;">Dear {name},</p>
                
                <p style="color: #34495e;">Thank you for choosing FinDoc.ai! We're excited to have you on board. Your account has been successfully created and you're all set to start using our intelligent document processing services.</p>
                
                <div style="background-color: #ffffff; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <h3 style="color: #2c3e50; margin-top: 0;">Quick Start Guide:</h3>
                    <ul style="color: #34495e;">
                        <li>Upload your documents securely</li>
                        <li>Choose between Quick or Premium processing</li>
                        <li>Get instant analysis and insights</li>
                        <li>Access your documents anywhere, anytime</li>
                    </ul>
                </div>

                <p style="color: #34495e;">We value your feedback! Help us improve by sharing your experience:</p>
                
                <div style="text-align: center; margin: 25px 0;">
                    <a href="https://docs.google.com/forms/d/e/1FAIpQLSfA_v4mWcH4LI2jb-axOXexXqvL7tUbAH9xcnDCpD2NMOyAww/viewform?usp=dialog" 
                       style="background-color: #3498db; 
                              color: white; 
                              padding: 12px 25px; 
                              text-decoration: none; 
                              border-radius: 5px;
                              font-weight: bold;">
                        Share Your Feedback
                    </a>
                </div>

                <p style="color: #34495e;">Need help? Our support team is always here to assist you. Simply reply to this email!</p>

                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee;">
                    <p style="color: #7f8c8d; margin: 0;">Best regards,</p>
                    <p style="color: #2c3e50; font-weight: bold; margin: 5px 0;">The FinDoc.ai Team</p>
                </div>
                
                <div style="margin-top: 20px; font-size: 12px; color: #7f8c8d;">
                    <p>Follow us on social media:</p>
                    <a href="#" style="color: #3498db; text-decoration: none; margin-right: 10px;">Twitter</a>
                    <a href="#" style="color: #3498db; text-decoration: none; margin-right: 10px;">LinkedIn</a>
                    <a href="#" style="color: #3498db; text-decoration: none;">Facebook</a>
                </div>
            </div>
        </body>
    </html>
    """
    
    email_service = EmailService()
    
    result = email_service.send_email(
        recipient_email="soumyadip.paik.gns@gmail.com",
        subject="Welcome to FinDoc.ai",
        html_content=html_template.format(name="Paik")
    )
    
    print(result["message"]) 