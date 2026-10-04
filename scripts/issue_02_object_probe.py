"""Run with a Python environment containing SB3 dependencies and --source fixed checkout."""
import argparse, json, subprocess, sys
from pathlib import Path
BASELINE="7cfb4dd6055e74b5caa4ed4d6777492209946e26"
p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); a=p.parse_args(); src=a.source.resolve()
assert subprocess.check_output(["git","-C",str(src),"rev-parse","HEAD"],text=True).strip()==BASELINE
sys.path.insert(0,str(src))
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.ppo.policies import MlpPolicy
class Trace(BaseCallback):
 def __init__(self,stop=False): super().__init__(); self.stop=stop; self.events=[]
 def _on_training_start(self): self.events.append("start")
 def _on_rollout_start(self): self.events.append("rollout_start")
 def _on_step(self): self.events.append("step"); assert self.training_env is self.model.env; return not self.stop
 def _on_rollout_end(self): self.events.append("rollout_end")
 def _on_training_end(self): self.events.append("end")
class DictObs(gym.ObservationWrapper):
 def __init__(self,e): super().__init__(e); self.observation_space=gym.spaces.Dict({"state":e.observation_space})
 def observation(self,o): return {"state":o}
m=PPO("MlpPolicy",gym.make("CartPole-v1"),n_steps=8,batch_size=4,n_epochs=1,seed=2,device="cpu"); env=m.get_env(); pol,buf=m.policy,m.rollout_buffer
assert isinstance(env,DummyVecEnv) and isinstance(env,VecEnv) and not issubclass(VecEnv,gym.Env) and MlpPolicy is ActorCriticPolicy and isinstance(buf,RolloutBuffer)
pos=buf.pos; m.predict(env.reset()); assert buf.pos==pos
t=Trace(); m.learn(16,callback=t); assert m.policy is pol and m.rollout_buffer is buf and m._n_updates==2 and t.n_calls==16
s=Trace(True); m.learn(8,callback=s); assert s.events==["start","rollout_start","step","end"] and buf.pos==0
d=PPO("MultiInputPolicy",DictObs(gym.make("CartPole-v1")),n_steps=8,batch_size=4,n_epochs=1,seed=2,device="cpu"); assert isinstance(d.rollout_buffer,DictRolloutBuffer); d.learn(8)
print(json.dumps({"commit":BASELINE,"ppo_mro":[x.__name__ for x in PPO.__mro__],"policy":type(pol).__name__,"buffer":type(buf).__name__,"env":type(env).__name__,"callback_events":t.events,"early_stop":s.events,"dict_buffer":type(d.rollout_buffer).__name__,"result":"PASS"},indent=2))
