ffrom fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from pydantic import BaseModel

app = FastAPI()




@app.get("/")
async def root():
    return {"message": "Hey, I'm here!"}


from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from pydantic import BaseModel

app = FastAPI()

class ScoringItem(BaseModel):
    Age: int
    Gender: int
    self_employed: int
    family_history: int
    work_interfere: int
    no_employees: int
    remote_work: int
    tech_company: int
    benefits: int
    care_options: int
    wellness_program: int
    seek_help: int
    anonymity: int
    leave: int
    mental_health_consequence: int
    phys_health_consequence: int
    coworkers: int
    supervisor: int
    mental_health_interview: int
    phys_health_interview: int
    mental_vs_physical: int
    obs_consequence: int

@app.post('/pred')
async def scoring_endpoint(item: ScoringItem):
    with open('main_model_git.pkl', 'rb') as f:
        model = joblib.load(f)

    data = item.dict()
    df = pd.DataFrame([data])
    yhat = model.predict(df)
    return {"prediction": int(yhat)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
