function [weights, rewards, policy] = runArsWithBias(env, alpha, sigma, nDelta, nTop, N)

weights = zeros(env.getObservationInfo.Dimension(1), env.getActionInfo.Dimension(1) + 1);
rewards = zeros(N,1);

policy_std = ones(size(weights,1),1);   
totalT = 0;

var_thresh = 1e-6;

for epoch = 1:N
    deltas = randn(size(weights,1), size(weights,2), nDelta)*sigma;   
    candidateWeights = cat(3, weights+deltas, weights-deltas);    
    
    
    Rs = zeros(1, size(candidateWeights,3));  
    Xstds = ones(size(weights,1), size(candidateWeights,3));  
    rolloutTs = zeros(1, size(candidateWeights,3));  
    
    begin = tic;
    
    candidateLinWeights = candidateWeights(:,1:end-1,:);
    candidateBiasWeights = candidateWeights(:,end,:);
    parfor i = 1:size(candidateWeights,3)
        W = candidateLinWeights(:,i); 
        mu = candidateBiasWeights(:,i);
        policy = @(x)(W'*((x - mu)./policy_std));
        [R,X,T] = doArsRollout(policy,env); 
        Rs(i) = R;
        Xstds(:,i) = std(X,0,2);
        rolloutTs(i) = T;
    end
    
    % update reward std ---------------------------------------------------
    for i = 1:size(candidateWeights,3)
        if any(Xstds(:,i) < var_thresh)
           continue 
        end
        cur_var = policy_std.^2;   
        new_var = Xstds(:,i).^2; 
        
        updated_var = (cur_var*totalT + new_var*rolloutTs(i))./(totalT + rolloutTs(i)); 
        policy_std = sqrt(updated_var);
    end

    totalT = totalT + sum(rolloutTs);
    
    rewardsPlus = Rs(1:nDelta);
    rewardsMinus = Rs(nDelta+1:end); 
    
    rewardsMax = max(rewardsPlus, rewardsMinus);  
    
    [sortedMaxRewards, sortedRewardsIdx] = sort(rewardsMax,'descend'); 
    Rdiff = (rewardsPlus - rewardsMinus);
    stepSize = alpha/(size(deltas,1)*std(Rs) + 1e-6);
    
    
    sortedRdiff = Rdiff(sortedRewardsIdx(1:nTop));
    sortedDeltas = deltas(:,:,sortedRewardsIdx(1:nTop)); 
    step = zeros(size(candidateWeights,1), size(candidateWeights,2)); 
    
    for j = 1:nTop
       step = step + sortedRdiff(j).*sortedDeltas(:,:,j);
    end
       
    weights = weights + stepSize*step;
    rewards(epoch) = mean(sortedMaxRewards(1:nTop));
    
    
    if mod(epoch, 10) == 0
        fprintf("iteration %d, FPS: %f \n", epoch, sum(rolloutTs)/toc(begin));
        fprintf("max(rewardsMax): %f, mean(): %f \n\n", max(rewardsMax), mean(rewardsMax));
    end
 
    
    
end

policy = @(x)(weights(:,1:end-1)'*((x - weights(:,end))./policy_std));
end