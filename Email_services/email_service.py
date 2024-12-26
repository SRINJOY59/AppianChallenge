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
    <html>
        <body>
            <h2>Welcome to FinDoc.ai!</h2>
            <p>Hello {name},</p>
            <p>Thank you for using our service. Your account has been successfully created.</p>
            <p>Best regards,<br>Your Team</p>
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