"""Email notification module for sending screening alerts.

Supports Gmail, Outlook, and custom SMTP servers with HTML email formatting.
"""

import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmailNotifier:
    """Send screening results via email with HTML formatting.

    Supports Gmail, Outlook, and custom SMTP servers. Uses environment
    variables for configuration.

    Environment Variables:
        EMAIL_FROM: Sender email address
        EMAIL_PASSWORD: Sender email password or app-specific password
        EMAIL_TO: Recipient email address (comma-separated for multiple)
        SMTP_SERVER: SMTP server (default: smtp.gmail.com for Gmail)
        SMTP_PORT: SMTP port (default: 587)

    Example:
        >>> notifier = EmailNotifier()
        >>> notifier.send_screening_results(results_df, top_n=10)
    """

    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
        email_from: Optional[str] = None,
        email_password: Optional[str] = None,
        email_to: Optional[str] = None
    ) -> None:
        """Initialize the email notifier.

        Args:
            smtp_server: SMTP server address. Defaults to env EMAIL_SMTP_SERVER or smtp.gmail.com.
            smtp_port: SMTP port. Defaults to env EMAIL_SMTP_PORT or 587.
            email_from: Sender email. Defaults to env EMAIL_FROM.
            email_password: Sender password. Defaults to env EMAIL_PASSWORD.
            email_to: Recipient email(s). Defaults to env EMAIL_TO.
        """
        self.smtp_server = smtp_server or os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = smtp_port or int(os.getenv('EMAIL_SMTP_PORT', '587'))
        self.email_from = email_from or os.getenv('EMAIL_FROM')
        self.email_password = email_password or os.getenv('EMAIL_PASSWORD')
        self.email_to = email_to or os.getenv('EMAIL_TO')

        if not self.email_from or not self.email_password:
            logger.warning("Email credentials not configured. Set EMAIL_FROM and EMAIL_PASSWORD.")

        logger.info(f"EmailNotifier initialized (SMTP: {self.smtp_server}:{self.smtp_port})")

    def _format_html_table(self, df: pd.DataFrame) -> str:
        """Format DataFrame as HTML table with styling.

        Args:
            df: DataFrame to format.

        Returns:
            HTML string with styled table.
        """
        html = '<table style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;">\n'

        # Header
        html += '  <thead>\n    <tr style="background-color: #2c3e50; color: white;">\n'
        for col in df.columns:
            html += f'      <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">{col}</th>\n'
        html += '    </tr>\n  </thead>\n'

        # Body
        html += '  <tbody>\n'
        for idx, row in df.iterrows():
            bg_color = '#f9f9f9' if idx % 2 == 0 else 'white'
            html += f'    <tr style="background-color: {bg_color};">\n'

            for col in df.columns:
                value = row[col]

                # Format based on column type
                if pd.isna(value):
                    display = 'N/A'
                elif col in ['buy_signal', 'value_score', 'support_score']:
                    # Color code scores
                    val = float(value)
                    if val >= 80:
                        color = '#27ae60'  # Green
                    elif val >= 65:
                        color = '#f39c12'  # Orange
                    else:
                        color = '#95a5a6'  # Gray
                    display = f'<span style="color: {color}; font-weight: bold;">{val:.1f}</span>'
                elif col == 'current_price':
                    display = f'${float(value):.2f}'
                elif col == 'rsi':
                    val = float(value)
                    if val < 30:
                        color = '#e74c3c'  # Red (oversold)
                    elif val < 70:
                        color = '#000'  # Black
                    else:
                        color = '#c0392b'  # Dark red (overbought)
                    display = f'<span style="color: {color};">{val:.1f}</span>'
                elif isinstance(value, (int, float)):
                    display = f'{value:.2f}'
                else:
                    display = str(value)

                html += f'      <td style="padding: 10px; border: 1px solid #ddd;">{display}</td>\n'

            html += '    </tr>\n'

        html += '  </tbody>\n</table>'
        return html

    def _create_html_email(
        self,
        results: pd.DataFrame,
        top_n: int = 10,
        subject_prefix: str = "[Stock Screener]"
    ) -> str:
        """Create HTML email body with screening results.

        Args:
            results: DataFrame with screening results.
            top_n: Number of top candidates to include.
            subject_prefix: Prefix for email subject line.

        Returns:
            HTML email body as string.
        """
        today = datetime.now().strftime('%B %d, %Y')
        total_candidates = len(results)

        # Get top candidates
        top_results = results.head(top_n)

        # Select columns for email
        email_cols = [
            'ticker', 'buy_signal', 'value_score', 'support_score',
            'current_price', 'rsi', 'pe_ratio', 'pb_ratio'
        ]
        display_df = top_results[email_cols].copy()

        # Rename columns for display
        display_df.columns = [
            'Ticker', 'Buy Signal', 'Value', 'Support',
            'Price', 'RSI', 'P/E', 'P/B'
        ]

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .summary {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #667eea;
        }}
        .legend {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #ffc107;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #eee;
            font-size: 12px;
            color: #666;
        }}
        .signal-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 12px;
        }}
        .strong-buy {{ background-color: #d4edda; color: #155724; }}
        .buy {{ background-color: #fff3cd; color: #856404; }}
        .consider {{ background-color: #d1ecf1; color: #0c5460; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Daily Stock Screening Results</h1>
        <p style="margin: 10px 0 0 0; font-size: 16px;">{today}</p>
    </div>

    <div class="summary">
        <h2 style="margin-top: 0;">Summary</h2>
        <p><strong>{total_candidates}</strong> stocks screened today. Showing top <strong>{len(top_results)}</strong> candidates.</p>
        <p>
            <span class="signal-badge strong-buy">🔥 STRONG BUY (80+)</span>
            <span class="signal-badge buy">✅ BUY (65-79)</span>
            <span class="signal-badge consider">⚡ CONSIDER (50-64)</span>
        </p>
    </div>

    <h2>Top {len(top_results)} Candidates</h2>

    {self._format_html_table(display_df)}

    <div class="legend">
        <h3 style="margin-top: 0;">📈 What These Scores Mean</h3>
        <ul style="margin: 10px 0;">
            <li><strong>Buy Signal:</strong> Combined score (70+ is actionable)</li>
            <li><strong>Value Score:</strong> Fundamental valuation (80+ is excellent)</li>
            <li><strong>Support Score:</strong> Technical setup (80+ is ready to buy)</li>
            <li><strong>RSI:</strong> <30 = Oversold (buy opportunity), >70 = Overbought</li>
        </ul>
    </div>

    <div class="footer">
        <p><strong>Automated Stock Screener</strong></p>
        <p>This email was automatically generated by your stock screening system.</p>
        <p>⚠️ This is not financial advice. Always do your own research before investing.</p>
    </div>
</body>
</html>
"""
        return html

    def send_screening_results(
        self,
        results: pd.DataFrame,
        top_n: int = 10,
        subject_prefix: str = "[Stock Screener]"
    ) -> bool:
        """Send screening results via email.

        Args:
            results: DataFrame with screening results (from screen_candidates).
            top_n: Number of top candidates to include in email.
            subject_prefix: Prefix for email subject line.

        Returns:
            True if email sent successfully, False otherwise.

        Example:
            >>> notifier = EmailNotifier()
            >>> results = screen_candidates(db, tickers)
            >>> notifier.send_screening_results(results, top_n=10)
        """
        if not self.email_from or not self.email_password or not self.email_to:
            logger.error("Email configuration incomplete. Check environment variables.")
            return False

        if results.empty:
            logger.warning("No screening results to send")
            return False

        try:
            # Create message
            today = datetime.now().strftime('%b %d, %Y')
            subject = f"{subject_prefix} Top {top_n} Candidates - {today}"

            msg = MIMEMultipart('alternative')
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            msg['Subject'] = subject

            # Create HTML body
            html_body = self._create_html_email(results, top_n, subject_prefix)

            # Create plain text fallback
            text_body = self._create_text_fallback(results, top_n)

            # Attach both versions
            msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))

            # Send email
            logger.info(f"Connecting to SMTP server: {self.smtp_server}:{self.smtp_port}")
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_from, self.email_password)

                recipients = [r.strip() for r in self.email_to.split(',')]
                server.sendmail(self.email_from, recipients, msg.as_string())

            logger.info(f"✓ Email sent successfully to {self.email_to}")
            return True

        except smtplib.SMTPAuthenticationError:
            logger.error("SMTP authentication failed. Check email credentials.")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def _create_text_fallback(self, results: pd.DataFrame, top_n: int) -> str:
        """Create plain text version of email for clients that don't support HTML.

        Args:
            results: DataFrame with screening results.
            top_n: Number of top candidates.

        Returns:
            Plain text email body.
        """
        today = datetime.now().strftime('%B %d, %Y')
        text = f"DAILY STOCK SCREENING RESULTS - {today}\n"
        text += "=" * 60 + "\n\n"

        text += f"Found {len(results)} candidates. Top {top_n} below:\n\n"

        # Format as table
        top_results = results.head(top_n)
        text += f"{'Ticker':<8} {'Buy Signal':<12} {'Value':<8} {'Support':<10} {'Price':<10}\n"
        text += "-" * 60 + "\n"

        for _, row in top_results.iterrows():
            text += f"{row['ticker']:<8} "
            text += f"{row['buy_signal']:<12.1f} "
            text += f"{row['value_score']:<8.1f} "
            text += f"{row['support_score']:<10.1f} "
            text += f"${row['current_price']:<9.2f}\n"

        text += "\n" + "=" * 60 + "\n"
        text += "\nLegend:\n"
        text += "- Buy Signal: Combined score (70+ is actionable)\n"
        text += "- Value Score: Fundamental valuation (80+ is excellent)\n"
        text += "- Support Score: Technical setup (80+ is ready to buy)\n"

        text += "\n" + "=" * 60 + "\n"
        text += "\n⚠️ This is not financial advice. Always do your own research.\n"

        return text

    def test_connection(self) -> bool:
        """Test SMTP connection and authentication.

        Returns:
            True if connection successful, False otherwise.

        Example:
            >>> notifier = EmailNotifier()
            >>> if notifier.test_connection():
            ...     print("Email configuration is valid!")
        """
        if not self.email_from or not self.email_password:
            logger.error("Email credentials not configured")
            return False

        try:
            logger.info(f"Testing connection to {self.smtp_server}:{self.smtp_port}")
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.email_from, self.email_password)
                logger.info("✓ SMTP connection successful")
                return True
        except smtplib.SMTPAuthenticationError:
            logger.error("✗ Authentication failed. Check email and password.")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"✗ SMTP error: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Connection failed: {e}")
            return False
