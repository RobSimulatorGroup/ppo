from pathlib import Path


LQR_PLAYBACK = Path("/home/wqq/gobot/ppo/scripts/cartpole_lqr_playback.py")

exec(compile(LQR_PLAYBACK.read_text(), str(LQR_PLAYBACK), "exec"))
