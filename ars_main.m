env = rlPredefinedEnv('CartPole-Continuous');

nDelta = 32;
nTop = 32;
stepSize = .36;
deltaStd = .05;
nEpochs = 1000;

begin = tic;
agent = ARSAgent(env, stepSize, deltaStd, nDelta, nTop, useBias=true);

rewards = agent.learn(nEpochs, verbose=1);

fprintf("Avg Episodes Per Second: %f \n",  nEpochs*2*nDelta/toc(begin));

plot(rewards);

[R,X] = doArsRollout(agent.policy, agent.env);
figure()
plot(X);