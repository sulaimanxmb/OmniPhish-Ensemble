FROM mcr.microsoft.com/playwright/python:v1.42.0-jammy

WORKDIR /app

# Install Python requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# The Microsoft Playwright image already creates a user with UID 1000 (usually 'pwuser')
# Hugging Face Spaces requires running as UID 1000, so we just switch to it directly.
USER 1000

# Copy application files with correct permissions
COPY --chown=1000 . /app/

# Expose Gradio port
EXPOSE 7860

# Run the Gradio App
CMD ["python", "app.py"]
