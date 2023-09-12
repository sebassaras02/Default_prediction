from pydantic import BaseModel

# define values that PredictionRequest that will receive


class PredictionRequest(BaseModel):
    Gender : str 
    Credit_Score : int    
    loan_purpose : str    
    loan_amount : int
    rate_of_interest : float
    age : str 
    Region : str
    
class PredictionResponse(BaseModel):
    status : int