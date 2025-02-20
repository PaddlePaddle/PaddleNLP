import paddle

class MoeConfig:
    def __init__(self):
        self.top_k = 5
        self.routed_scaling_factor = 0.1

class Model:
    def __init__(self):
        self.config = MoeConfig()

    def forward(self, scores):
        # Step 1: Perform top-k operation
        _, topk_ids = paddle.topk(scores, k=self.config.top_k, axis=-1, sorted=False)
        
        print(topk_ids)
        # Step 2: Create new_mask with the top-k indices

        new_mask = paddle.zeros_like(scores, dtype="int64")
        new_mask = paddle.put_along_axis(new_mask, topk_ids, 1, axis=1)
        
        print(new_mask)
        # Step 3: Apply the mask to the scores
        scores = scores * new_mask.astype("float32")

        # Step 4: Normalize the scores
        score_sum = paddle.sum(scores, axis=-1, keepdim=True) + 1e-20
        scores = scores / score_sum * self.config.routed_scaling_factor

        return scores


# Test the code
paddle.set_device('cpu')  # Assuming we are using CPU, change to 'gpu' if applicable
model = Model()

# Create a random tensor of scores
batch_size = 2
num_classes = 10
scores = paddle.randn([batch_size, num_classes], dtype='float32')

print("Input Scores:")
print(scores)

# Forward pass
output_scores = model.forward(scores)

print("\nOutput Scores after processing:")
print(output_scores)
