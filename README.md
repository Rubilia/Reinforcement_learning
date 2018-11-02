# Reinforcement_learning
In this repository you can find implementations of major RL algorithms. Those algorithms are applied to different environments.
<h1>Random Walk</h1>
<p style="margin-left: 40px">Imagine you have n circles drawn in the one line one after another.</p>
<img src="https://www.researchgate.net/profile/L_Dinis/publication/243403140/figure/fig2/AS:298466971537414@1448171479541/figure-fig2_W840.webp"></img>
<p>An agent appers in random circle(state) at the beginig of episode. Each circle has unique number. Let State s be the number of n-th circle
Agent can go left or right, decrease or increse state by 1. When agent leaves environment from left state, it earns some negative reward, if it leaves in the right state, it earns some positive reward.
Otherwise it earns nothing (zero reward). Agents target is to learn optimal behavior to maximize incoming rewards. In this case agent should always go right.
There`re 3 algorithms solves this problem: Monte Carlo method, TD(0), TD(lambda).
</p>
