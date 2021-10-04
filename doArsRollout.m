function [R,X,T] = doArsRollout(policy,env)
% Does a rollout for ars given a policy function from observation->actions,
% and an environment object that implements Matlabs RL API

x = env.reset();
X = [];
isDone = false;
R = 0;

for step = 1:1000
   a = policy(x);
   [x,r,isDone,~] = env.step(a);
   X = [X, x];
   R = R + r;
   if isDone
       break
   end
end

T = size(X,2);

end
