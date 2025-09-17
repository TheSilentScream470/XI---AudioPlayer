# fastapi_main.py
import os
import platform
import sys
import io
import shutil
import random
import asyncio
import tempfile
import threading
import traceback
import subprocess
import uuid
import stat
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import time
import json
import hashlib
import secrets

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Query, Cookie, Response, Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, field_validator, model_validator, EmailStr
import aiofiles
from mutagen import File as MutagenFile
from mutagen.id3 import ID3, APIC
from pydub import AudioSegment
import yt_dlp
import requests
from urllib.parse import urlparse
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="XI Audio Player API", version="1.0.0")

# Security
security = HTTPBasic()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories with proper permissions
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
MUSIC_DIR = BASE_DIR / "music"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
USER_DATA_DIR = BASE_DIR / "user_data"
USERS_DIR = BASE_DIR / "users"

for directory in [CACHE_DIR, MUSIC_DIR, STATIC_DIR, USER_DATA_DIR, USERS_DIR]:
    directory.mkdir(exist_ok=True)
    # Set proper permissions (read/write for everyone)
    try:
        os.chmod(directory, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    except:
        pass  # Skip if permission change fails

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Thread pool for background tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

# User session management
user_states = {}  # session_id -> PlayerState
user_sessions = {}  # session_id -> user_id

# Pydantic models
class TrackInfo(BaseModel):
    index: int
    title: str
    artist: Optional[str] = None
    album: Optional[str] = None
    duration: Optional[float] = None
    format: Optional[str] = None
    size: Optional[float] = None
    path: str

class PlayerStatus(BaseModel):
    is_playing: bool
    current_track: Optional[TrackInfo] = None
    volume: int
    speed: float
    position: int
    playlist_length: int

    @model_validator(mode='before')
    @classmethod
    def validate_position(cls, data):
        if isinstance(data, dict) and 'position' in data:
            # Convert float position to int
            data['position'] = int(data['position'])
        return data

class PlaylistResponse(BaseModel):
    tracks: List[TrackInfo]
    current_index: int

class UserRegister(BaseModel):
    username: str
    password: str
    email: EmailStr

class UserLogin(BaseModel):
    username: str
    password: str

class FriendRequest(BaseModel):
    friend_username: str

class SharePlaylistRequest(BaseModel):
    friend_username: str

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# User management functions
def hash_password(password: str) -> str:
    """Hash a password for storing."""
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}${pwd_hash}"

def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verify a stored password against one provided by user"""
    salt, stored_hash = stored_password.split('$')
    pwd_hash = hashlib.sha256((provided_password + salt).encode()).hexdigest()
    return pwd_hash == stored_hash

def get_user_file(username: str) -> Path:
    """Get the user data file path"""
    return USERS_DIR / f"{username}.json"

def load_user_data(username: str) -> Optional[Dict]:
    """Load user data from file"""
    user_file = get_user_file(username)
    if user_file.exists():
        with open(user_file, 'r') as f:
            return json.load(f)
    return None

def save_user_data(username: str, data: Dict):
    """Save user data to file"""
    user_file = get_user_file(username)
    with open(user_file, 'w') as f:
        json.dump(data, f)

def create_user(username: str, password: str, email: str) -> bool:
    """Create a new user"""
    if get_user_file(username).exists():
        return False
    
    user_data = {
        "username": username,
        "password_hash": hash_password(password),
        "email": email,
        "friends": [],
        "friend_requests": [],
        "shared_playlists": {},
        "created_at": time.time()
    }
    
    save_user_data(username, user_data)
    return True

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user"""
    user_data = load_user_data(username)
    if not user_data:
        return False
    
    return verify_password(user_data["password_hash"], password)

# Player state for each user
class PlayerState:
    def __init__(self, session_id: str, username: str):
        self.session_id = session_id
        self.username = username
        self.playlist = []
        self.current_index = -1
        self.is_playing = False
        self.current_speed = 1.0
        self.volume = 80
        self.playback_position = 0
        self.user_dir = USER_DATA_DIR / username
        self.user_dir.mkdir(exist_ok=True)
        
    def save_state(self):
        """Save user state to disk"""
        state_file = self.user_dir / "state.json"
        state_data = {
            "playlist": self.playlist,
            "current_index": self.current_index,
            "volume": self.volume,
            "speed": self.current_speed
        }
        with open(state_file, 'w') as f:
            json.dump(state_data, f)
    
    def load_state(self):
        """Load user state from disk"""
        state_file = self.user_dir / "state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state_data = json.load(f)
                self.playlist = state_data.get("playlist", [])
                self.current_index = state_data.get("current_index", -1)
                self.volume = state_data.get("volume", 80)
                self.current_speed = state_data.get("speed", 1.0)

def get_user_state(session_id: str, username: str) -> PlayerState:
    """Get or create user state for session"""
    if session_id not in user_states:
        user_states[session_id] = PlayerState(session_id, username)
        user_states[session_id].load_state()
    return user_states[session_id]

def get_current_user(session_id: str = Cookie(default=None)):
    """Get current user from session"""
    if not session_id or session_id not in user_sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user_sessions[session_id]

# Utility functions
def get_ffmpeg_path():
    """Get FFmpeg path with fallback to system PATH"""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return "ffmpeg"
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise HTTPException(status_code=500, detail="FFmpeg not found. Please install FFmpeg.")

def extract_thumbnail(file_path: str) -> Optional[bytes]:
    """Extract thumbnail from audio file"""
    try:
        audio_file = MutagenFile(file_path)
        if not audio_file:
            return None
            
        thumbnail_data = None
        
        # For MP3 (ID3 tags)
        if hasattr(audio_file, 'tags') and 'APIC:' in audio_file.tags:
            thumbnail_data = audio_file.tags['APIC:'].data
        # For MP4/M4A
        elif hasattr(audio_file, 'tags') and 'covr' in audio_file.tags:
            thumbnail_data = audio_file.tags['covr'][0]
        # For FLAC
        elif hasattr(audio_file, 'pictures') and audio_file.pictures:
            thumbnail_data = audio_file.pictures[0].data
        # For OGG/Vorbis
        elif hasattr(audio_file, 'tags') and 'metadata_block_picture' in audio_file.tags:
            from base64 import b64decode
            picture_data = audio_file.tags['metadata_block_picture'][0]
            from mutagen.flac import Picture
            picture = Picture(b64decode(picture_data))
            thumbnail_data = picture.data
            
        return thumbnail_data if thumbnail_data and len(thumbnail_data) > 100 else None
    except Exception as e:
        print(f"Error extracting thumbnail: {str(e)}")
        return None

def get_track_info(file_path: str) -> TrackInfo:
    """Get track information from file"""
    try:
        audio = MutagenFile(file_path)
        title = artist = album = None
        duration = format = size = None
        
        if audio and hasattr(audio, 'tags'):
            if 'TIT2' in audio.tags:
                title = audio.tags['TIT2'].text[0]
            if 'TPE1' in audio.tags:
                artist = audio.tags['TPE1'].text[0]
            if 'TALB' in audio.tags:
                album = audio.tags['TALB'].text[0]
                
        if hasattr(audio.info, 'length'):
            duration = audio.info.length
            
        format = Path(file_path).suffix[1:].upper() if Path(file_path).suffix else "UNKNOWN"
        size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        if not title:
            title = Path(file_path).stem
            
        return TrackInfo(
            index=-1,  # Will be set by caller
            title=title,
            artist=artist,
            album=album,
            duration=duration,
            format=format,
            size=size,
            path=file_path
        )
    except Exception as e:
        print(f"Error getting track info: {str(e)}")
        return TrackInfo(
            index=-1,
            title=Path(file_path).stem,
            path=file_path
        )

def scan_music_directory(directory: str) -> List[str]:
    """Scan directory for audio files"""
    supported_formats = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac")
    audio_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_formats):
                full_path = os.path.join(root, file)
                try:
                    if os.access(full_path, os.R_OK):
                        audio_files.append(full_path)
                except Exception as e:
                    print(f"Error checking file {full_path}: {str(e)}")
    
    return audio_files

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the web player interface"""
    return templates.TemplateResponse("player.html", {"request": request})

# Authentication endpoints
@app.post("/api/register")
async def register(user_data: UserRegister):
    """Register a new user"""
    if get_user_file(user_data.username).exists():
        raise HTTPException(status_code=400, detail="Username already exists")
    
    if create_user(user_data.username, user_data.password, user_data.email):
        return {"message": "User created successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to create user")

@app.post("/api/login")
async def login(response: Response, credentials: HTTPBasicCredentials = Depends(security)):
    """Login user and create session"""
    if not authenticate_user(credentials.username, credentials.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create session
    session_id = str(uuid.uuid4())
    user_sessions[session_id] = credentials.username
    response.set_cookie(key="session_id", value=session_id)
    
    # Initialize user state
    get_user_state(session_id, credentials.username)
    
    return {"message": "Login successful", "username": credentials.username}

@app.post("/api/logout")
async def logout(response: Response, session_id: str = Cookie(default=None)):
    """Logout user and clear session"""
    if session_id in user_sessions:
        # Save state before logout
        if session_id in user_states:
            user_states[session_id].save_state()
            del user_states[session_id]
        
        del user_sessions[session_id]
    
    response.delete_cookie(key="session_id")
    return {"message": "Logout successful"}

@app.get("/api/user/profile")
async def get_profile(username: str = Depends(get_current_user)):
    """Get user profile"""
    user_data = load_user_data(username)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Return profile without sensitive data
    return {
        "username": user_data["username"],
        "email": user_data["email"],
        "friends": user_data.get("friends", []),
        "friend_requests": user_data.get("friend_requests", []),
        "created_at": user_data.get("created_at", 0)
    }

# Friends endpoints
@app.post("/api/friends/request")
async def send_friend_request(
    request: FriendRequest, 
    username: str = Depends(get_current_user)
):
    """Send a friend request"""
    friend_data = load_user_data(request.friend_username)
    if not friend_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    if request.friend_username == username:
        raise HTTPException(status_code=400, detail="Cannot add yourself as friend")
    
    if request.friend_username in friend_data.get("friends", []):
        raise HTTPException(status_code=400, detail="Already friends")
    
    # Add to friend's pending requests
    if "friend_requests" not in friend_data:
        friend_data["friend_requests"] = []
    
    if username not in friend_data["friend_requests"]:
        friend_data["friend_requests"].append(username)
        save_user_data(request.friend_username, friend_data)
    
    return {"message": f"Friend request sent to {request.friend_username}"}

@app.post("/api/friends/accept/{friend_username}")
async def accept_friend_request(
    friend_username: str,
    username: str = Depends(get_current_user)
):
    """Accept a friend request"""
    user_data = load_user_data(username)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    if friend_username not in user_data.get("friend_requests", []):
        raise HTTPException(status_code=400, detail="No friend request from this user")
    
    # Remove from requests and add to friends
    user_data["friend_requests"].remove(friend_username)
    if "friends" not in user_data:
        user_data["friends"] = []
    if friend_username not in user_data["friends"]:
        user_data["friends"].append(friend_username)
    
    # Also add current user to friend's friends list
    friend_data = load_user_data(friend_username)
    if friend_data:
        if "friends" not in friend_data:
            friend_data["friends"] = []
        if username not in friend_data["friends"]:
            friend_data["friends"].append(username)
        save_user_data(friend_username, friend_data)
    
    save_user_data(username, user_data)
    
    return {"message": f"Accepted friend request from {friend_username}"}

@app.post("/api/friends/reject/{friend_username}")
async def reject_friend_request(
    friend_username: str,
    username: str = Depends(get_current_user)
):
    """Reject a friend request"""
    user_data = load_user_data(username)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    if friend_username in user_data.get("friend_requests", []):
        user_data["friend_requests"].remove(friend_username)
        save_user_data(username, user_data)
    
    return {"message": f"Rejected friend request from {friend_username}"}

@app.get("/api/friends")
async def get_friends(username: str = Depends(get_current_user)):
    """Get user's friends list"""
    user_data = load_user_data(username)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "friends": user_data.get("friends", []),
        "friend_requests": user_data.get("friend_requests", [])
    }

# Playlist sharing
@app.post("/api/playlist/share")
async def share_playlist(
    request: SharePlaylistRequest,
    username: str = Depends(get_current_user)
):
    """Share current playlist with a friend"""
    friend_username = request.friend_username
    
    # Check if friend exists
    friend_data = load_user_data(friend_username)
    if not friend_data:
        raise HTTPException(status_code=404, detail="Friend not found")
    
    # Check if they are friends
    user_data = load_user_data(username)
    if friend_username not in user_data.get("friends", []):
        raise HTTPException(status_code=400, detail="You can only share with friends")
    
    # Get current user's playlist
    session_id = None
    for sid, uname in user_sessions.items():
        if uname == username:
            session_id = sid
            break
    
    if not session_id or session_id not in user_states:
        raise HTTPException(status_code=400, detail="No active playlist to share")
    
    user_state = user_states[session_id]
    
    # Add to friend's shared playlists
    if "shared_playlists" not in friend_data:
        friend_data["shared_playlists"] = {}
    
    friend_data["shared_playlists"][username] = {
        "tracks": user_state.playlist,
        "shared_at": time.time()
    }
    
    save_user_data(friend_username, friend_data)
    
    return {"message": f"Playlist shared with {friend_username}"}

@app.get("/api/playlist/shared")
async def get_shared_playlists(username: str = Depends(get_current_user)):
    """Get playlists shared with user"""
    user_data = load_user_data(username)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user_data.get("shared_playlists", {})

@app.post("/api/playlist/load_shared/{friend_username}")
async def load_shared_playlist(
    friend_username: str,
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
):
    """Load a shared playlist from a friend"""
    user_data = load_user_data(username)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    shared_playlists = user_data.get("shared_playlists", {})
    if friend_username not in shared_playlists:
        raise HTTPException(status_code=404, detail="No shared playlist from this user")
    
    # Load the shared playlist
    shared_data = shared_playlists[friend_username]
    user_state = get_user_state(session_id, username)
    user_state.playlist = shared_data["tracks"]
    user_state.current_index = -1
    user_state.save_state()
    
    return {"message": f"Loaded playlist shared by {friend_username}", "count": len(user_state.playlist)}

# Player endpoints
@app.get("/api/status")
async def get_status(
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
) -> PlayerStatus:
    """Get current player status"""
    user_state = get_user_state(session_id, username)
    current_track = None
    if user_state.current_index >= 0 and user_state.playlist:
        current_track = get_track_info(user_state.playlist[user_state.current_index])
        current_track.index = user_state.current_index
    
    return PlayerStatus(
        is_playing=user_state.is_playing,
        current_track=current_track,
        volume=user_state.volume,
        speed=user_state.current_speed,
        position=int(user_state.playback_position),
        playlist_length=len(user_state.playlist)
    )

@app.get("/api/playlist")
async def get_playlist(
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
) -> PlaylistResponse:
    """Get current playlist"""
    user_state = get_user_state(session_id, username)
    tracks = []
    for i, path in enumerate(user_state.playlist):
        track_info = get_track_info(path)
        track_info.index = i
        tracks.append(track_info)
    
    return PlaylistResponse(tracks=tracks, current_index=user_state.current_index)

@app.post("/api/playlist/load")
async def load_playlist(
    directory: str = Form(...), 
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
):
    """Load playlist from directory"""
    if not os.path.isdir(directory):
        raise HTTPException(status_code=400, detail="Directory does not exist")
    
    audio_files = scan_music_directory(directory)
    if not audio_files:
        raise HTTPException(status_code=404, detail="No audio files found in directory")
    
    user_state = get_user_state(session_id, username)
    user_state.playlist = audio_files
    user_state.current_index = -1
    user_state.save_state()
    
    return {"message": f"Loaded {len(audio_files)} tracks", "count": len(audio_files)}

@app.post("/api/playlist/shuffle")
async def shuffle_playlist(
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
):
    """Shuffle the current playlist"""
    user_state = get_user_state(session_id, username)
    if not user_state.playlist:
        raise HTTPException(status_code=400, detail="Playlist is empty")
    
    current_track = None
    if user_state.current_index >= 0:
        current_track = user_state.playlist[user_state.current_index]
    
    random.shuffle(user_state.playlist)
    
    new_index = -1
    if current_track and current_track in user_state.playlist:
        new_index = user_state.playlist.index(current_track)
    
    user_state.current_index = new_index
    user_state.save_state()
    
    return {"message": "Playlist shuffled"}

@app.post("/api/play/{index}")
async def play_track(
    index: int, 
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
):
    """Set current track for playback"""
    user_state = get_user_state(session_id, username)
    if not user_state.playlist:
        raise HTTPException(status_code=400, detail="Playlist is empty")
    
    if index < 0 or index >= len(user_state.playlist):
        raise HTTPException(status_code=400, detail="Invalid track index")
    
    user_state.current_index = index
    user_state.is_playing = True
    user_state.playback_position = 0
    user_state.save_state()
    
    file_path = user_state.playlist[index]
    track_info = get_track_info(file_path)
    track_info.index = index
    
    return {"message": f"Playing track {index}", "track": track_info.dict()}

@app.post("/api/play")
async def play(
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
):
    """Start or resume playback"""
    user_state = get_user_state(session_id, username)
    if not user_state.playlist:
        raise HTTPException(status_code=400, detail="Playlist is empty")
    
    if user_state.current_index < 0:
        # Start from beginning if no track selected
        return await play_track(0, session_id, username)
    
    user_state.is_playing = True
    user_state.save_state()
    return {"message": "Playback resumed"}

@app.post("/api/pause")
async def pause(
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
):
    """Pause playback"""
    user_state = get_user_state(session_id, username)
    user_state.is_playing = False
    user_state.save_state()
    return {"message": "Playback paused"}

@app.post("/api/stop")
async def stop(
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
):
    """Stop playback"""
    user_state = get_user_state(session_id, username)
    user_state.is_playing = False
    user_state.playback_position = 0
    user_state.save_state()
    
    return {"message": "Playback stopped"}

@app.post("/api/next")
async def next_track(
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
):
    """Play next track"""
    user_state = get_user_state(session_id, username)
    if not user_state.playlist:
        raise HTTPException(status_code=400, detail="Playlist is empty")
    
    next_index = (user_state.current_index + 1) % len(user_state.playlist)
    return await play_track(next_index, session_id, username)

@app.post("/api/previous")
async def previous_track(
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
):
    """Play previous track"""
    user_state = get_user_state(session_id, username)
    if not user_state.playlist:
        raise HTTPException(status_code=400, detail="Playlist is empty")
    
    prev_index = (user_state.current_index - 1) % len(user_state.playlist)
    return await play_track(prev_index, session_id, username)

@app.post("/api/volume")
async def set_volume(
    volume: int = Form(...), 
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
):
    """Set playback volume"""
    user_state = get_user_state(session_id, username)
    user_state.volume = volume
    user_state.save_state()
    return {"message": f"Volume set to {volume}%"}

@app.post("/api/speed")
async def set_speed(
    speed: float = Form(...), 
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
):
    """Set playback speed"""
    user_state = get_user_state(session_id, username)
    user_state.current_speed = speed
    user_state.save_state()
    return {"message": f"Speed set to {speed}x"}

@app.post("/api/seek")
async def seek(
    position: int = Form(...), 
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
):
    """Seek to position in current track"""
    user_state = get_user_state(session_id, username)
    user_state.playback_position = position
    user_state.save_state()
    return {"message": f"Seeked to position {position}"}

@app.get("/api/thumbnail/{index}")
async def get_thumbnail(
    index: int, 
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
):
    """Get thumbnail for track"""
    user_state = get_user_state(session_id, username)
    if index < 0 or index >= len(user_state.playlist):
        raise HTTPException(status_code=400, detail="Invalid track index")
    
    file_path = user_state.playlist[index]
    thumbnail_data = extract_thumbnail(file_path)
    
    if not thumbnail_data:
        # Return default thumbnail
        default_thumb = STATIC_DIR / "default-thumb.png"
        if not default_thumb.exists():
            # Create a default thumbnail
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (150, 150), color=(50, 50, 50))
            d = ImageDraw.Draw(img)
            d.text((50, 65), "No Image", fill=(200, 200, 200))
            img.save(default_thumb)
        
        return FileResponse(default_thumb)
    
    return StreamingResponse(io.BytesIO(thumbnail_data), media_type="image/jpeg")

@app.post("/api/upload")
async def upload_file(
    files: List[UploadFile] = File(...), 
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
):
    """Upload audio files"""
    user_state = get_user_state(session_id, username)
    user_music_dir = user_state.user_dir / "music"
    user_music_dir.mkdir(exist_ok=True)
    
    uploaded_files = []
    
    for file in files:
        # Validate file type
        allowed_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            continue  # Skip unsupported files
        
        # Save file
        file_path = user_music_dir / file.filename
        
        # Ensure unique filename
        counter = 1
        original_stem = file_path.stem
        while file_path.exists():
            file_path = user_music_dir / f"{original_stem}_{counter}{file_path.suffix}"
            counter += 1
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Add to playlist
        user_state.playlist.append(str(file_path))
        uploaded_files.append(str(file_path))
    
    user_state.save_state()
    
    return {
        "message": f"Uploaded {len(uploaded_files)} files successfully", 
        "uploaded_files": uploaded_files,
        "failed_count": len(files) - len(uploaded_files)
    }

@app.post("/api/download")
async def download_from_url(
    url: str = Form(...), 
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
):
    """Download audio from URL (YouTube or direct link)"""
    try:
        if "youtube.com" in url or "youtu.be" in url:
            result = await download_youtube_audio(url, session_id, username)
        else:
            result = await download_audio_from_url(url, session_id, username)
        
        # Add to playlist
        user_state = get_user_state(session_id, username)
        user_state.playlist.append(result['path'])
        user_state.save_state()
        
        return {"message": "Audio downloaded successfully", "track": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

async def download_youtube_audio(url: str, session_id: str, username: str) -> Dict[str, Any]:
    """Download audio from YouTube"""
    loop = asyncio.get_event_loop()
    user_state = get_user_state(session_id, username)
    
    def download_task():
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': str(user_state.user_dir / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            temp_path = ydl.prepare_filename(info)
            temp_path = temp_path.rsplit('.', 1)[0] + '.mp3'
            
            # Get thumbnail
            thumbnail_url = info.get('thumbnail')
            thumbnail_data = None
            if thumbnail_url:
                try:
                    response = requests.get(thumbnail_url, timeout=10)
                    if response.status_code == 200:
                        thumbnail_data = response.content
                except:
                    pass
            
            return {
                'path': temp_path,
                'title': info.get('title', 'YouTube Audio'),
                'thumbnail': thumbnail_data
            }
    
    return await loop.run_in_executor(thread_pool, download_task)

async def download_audio_from_url(url: str, session_id: str, username: str) -> Dict[str, Any]:
    """Download audio from direct URL"""
    loop = asyncio.get_event_loop()
    user_state = get_user_state(session_id, username)
    
    def download_task():
        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                # Get filename from URL or content-disposition
                filename = os.path.basename(urlparse(url).path)
                if not filename or '.' not in filename:
                    filename = "audio.mp3"
                
                temp_path = os.path.join(user_state.user_dir, filename)
                
                # Ensure unique filename
                counter = 1
                original_stem = Path(temp_path).stem
                while os.path.exists(temp_path):
                    temp_path = os.path.join(user_state.user_dir, f"{original_stem}_{counter}{Path(temp_path).suffix}")
                    counter += 1
                
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                return {
                    'path': temp_path,
                    'title': os.path.splitext(filename)[0]
                }
            else:
                raise Exception(f"HTTP Error: {response.status_code}")
        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")
    
    return await loop.run_in_executor(thread_pool, download_task)

@app.post("/api/convert")
async def convert_audio(
    background_tasks: BackgroundTasks,
    input_path: str = Form(...),
    output_path: str = Form(...),
    bitrate: str = Form("320k"),
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
):
    """Convert audio to different bitrate"""
    if not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail="Input file does not exist")
    
    user_state = get_user_state(session_id, username)
    user_output_dir = user_state.user_dir / "converted"
    user_output_dir.mkdir(exist_ok=True)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Run conversion in background
    background_tasks.add_task(run_conversion, input_path, output_path, bitrate)
    
    return {"message": "Conversion started", "output_path": output_path}

def run_conversion(input_path: str, output_path: str, bitrate: str):
    """Run audio conversion (blocking)"""
    try:
        ffmpeg_path = get_ffmpeg_path()
        
        command = [
            ffmpeg_path,
            "-i", input_path,
            "-map", "0",
            "-c:a", "libmp3lame",
            "-b:a", bitrate,
            "-c:v", "copy",
            "-id3v2_version", "3",
            output_path
        ]
        
        result = subprocess.run(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
    except Exception as e:
        print(f"Conversion error: {str(e)}")

def change_speed(sound: AudioSegment, speed: float) -> AudioSegment:
    """Speed up or slow down with pitch shift (chipmunk/demon effect)."""
    altered = sound._spawn(
        sound.raw_data,
        overrides={"frame_rate": int(sound.frame_rate * speed)}
    )
    return altered.set_frame_rate(44100).set_channels(2)

@app.get("/api/stream/{index}")
async def stream_audio(
    index: int, 
    session_id: str = Cookie(default=None),
    username: str = Depends(get_current_user)
):
    """Stream audio with applied speed (pitch changes too)."""
    user_state = get_user_state(session_id, username)
    if index < 0 or index >= len(user_state.playlist):
        raise HTTPException(status_code=400, detail="Invalid track index")

    file_path = user_state.playlist[index]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # Load audio with pydub
        audio = AudioSegment.from_file(file_path)

        # Apply speed with pitch
        if user_state.current_speed != 1.0:
            audio = change_speed(audio, user_state.current_speed)

        # Export to memory
        buffer = io.BytesIO()
        audio.export(buffer, format="mp3")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="audio/mpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    # Save all user states
    for user_state in user_states.values():
        user_state.save_state()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)