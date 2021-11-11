function [R,X] = doArsRollout(policy, env, maxEnvSteps)
% Does a rollout for ars given a policy function from observation->actions,
% and an environment object that implements Matlabs RL API

arguments
   policy
   env
   maxEnvSteps = 1000;
    
end

x = env.reset();
X = zeros(maxEnvSteps, size(x,1));
R = 0;

for step = 1:maxEnvSteps
   a = policy(x);
   [x,r,isDone,~] = env.step(a);
   X(step,:) = x;
   R = R + r;
   if isDone
       X = X(1:step, :);
       break
   end
end

end
