# LAnguage-Modeling-Is-All-You-Need-for-Lifelong-Language-Learning
Most research on lifelong learning (LLL) applies to images or games, but not
language.
We present LAMAL, a simple yet effective method for LLL based on language
modeling.
LAMAL replays pseudo-samples of previous tasks while requiring no extra memory
or model capacity.
Specifically, LAMAL is a language model that simultaneously learns to solve the
task and generate training samples.
When the model is trained for a new task, it generates pseudo-samples of
previous tasks for training alongside data for the new task.
The results show that LAMAL prevents catastrophic forgetting without any sign of
intransigence and can perform up to five very different language tasks
sequentially with only one model. 
Overall, LAMAL outperforms previous methods by a considerable margin and is only
2--3\% worse than multitasking, which is usually considered the LLL upper bound.
