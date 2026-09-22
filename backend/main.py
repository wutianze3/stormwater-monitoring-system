from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from database import init_db
from routes.sensors import router as sensors_router
from routes.events import router as events_router
from routes.websocket import router as websocket_router, sensor_loop
from routes.vision import router as vision_router

app = FastAPI(title="Stormwater Monitoring System")

# allow requests from vue dev server during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# register all routers
app.include_router(sensors_router)
app.include_router(events_router)
app.include_router(websocket_router)
app.include_router(vision_router)

init_db()


@app.on_event("startup")
async def startup():
    # start the sensor loop as a background task on server startup
    asyncio.create_task(sensor_loop())
