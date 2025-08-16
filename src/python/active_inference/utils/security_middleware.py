
import logging
from functools import wraps

def security_headers(func):
    """Add security headers to responses."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)
        
        # Add security headers
        headers = {
            'X-Frame-Options': 'DENY',
            'X-Content-Type-Options': 'nosniff',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        }
        
        for header, value in headers.items():
            response.headers[header] = value
        
        return response
    return wrapper

def audit_log(action, user_id=None, details=None):
    """Log security-relevant actions."""
    logger = logging.getLogger('security.audit')
    logger.info({
        'action': action,
        'user_id': user_id,
        'details': details,
        'timestamp': datetime.now().isoformat()
    })
