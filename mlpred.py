from fastapi import FastAPI, Query, HTTPException
import pandas as pd
import joblib

app = FastAPI()




@app.get("/")
async def root():
    return {"message": "Hey, I'm here!"}



@app.get("/bro")
async def root():
    return {"message": "wassup"}


@app.get('/abc')
async def get_prediction(
    Age: int = Query(20, description="Age"),
    Gender: int = Query(1, description="Gender"),
    self_employed: int = Query(0, description="Self Employed"),
    family_history: int = Query(0, description="Family History"),
    work_interfere: int = Query(1, description="Work Interference"),
    no_employees: int = Query(0, description="No. of Employees"),
    remote_work: int = Query(1, description="Remote Work"),
    tech_company: int = Query(0, description="Tech Company"),
    benefits: int = Query(1, description="Benefits"),
    care_options: int = Query(0, description="Care Options"),
    wellness_program: int = Query(0, description="Wellness Program"),
    seek_help: int = Query(1, description="Seek Help"),
    anonymity: int = Query(0, description="Anonymity"),
    leave: int = Query(1, description="Leave"),
    mental_health_consequence: int = Query(0, description="Mental Health Consequence"),
    phys_health_consequence: int = Query(1, description="Physical Health Consequence"),
    coworkers: int = Query(0, description="Coworkers"),
    supervisor: int = Query(0, description="Supervisor"),
    mental_health_interview: int = Query(1, description="Mental Health Interview"),
    phys_health_interview: int = Query(0, description="Physical Health Interview"),
    mental_vs_physical: int = Query(1, description="Mental vs Physical"),
    obs_consequence: int = Query(0, description="Observed Consequence")
):
    # Check if any parameters are missing
  with open('main_model_git.pkl', 'rb') as f:
    model = joblib.load(f)  

    if (
        Age == Gender == self_employed == family_history == work_interfere == no_employees ==
        remote_work == tech_company == benefits == care_options == wellness_program == seek_help ==
        anonymity == leave == mental_health_consequence == phys_health_consequence == coworkers ==
        supervisor == mental_health_interview == phys_health_interview == mental_vs_physical ==
        obs_consequence == 0
    ):
        raise HTTPException(status_code=400, detail="No parameters provided.")

    data = {
        "Age": Age,
        "Gender": Gender,
        "self_employed": self_employed,
        "family_history": family_history,
        "work_interfere": work_interfere,
        "no_employees": no_employees,
        "remote_work": remote_work,
        "tech_company": tech_company,
        "benefits": benefits,
        "care_options": care_options,
        "wellness_program": wellness_program,
        "seek_help": seek_help,
        "anonymity": anonymity,
        "leave": leave,
        "mental_health_consequence": mental_health_consequence,
        "phys_health_consequence": phys_health_consequence,
        "coworkers": coworkers,
        "supervisor": supervisor,
        "mental_health_interview": mental_health_interview,
        "phys_health_interview": phys_health_interview,
        "mental_vs_physical": mental_vs_physical,
        "obs_consequence": obs_consequence,
    }

    df = pd.DataFrame([data])
    yhat = model.predict(df)

    return {"prediction": int(yhat)}
  
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)