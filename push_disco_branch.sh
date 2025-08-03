#!/bin/bash
# Script to push DISCO branch to remote repository

echo "Pushing DISCO branch to remote repository..."
echo "Current branch: $(git branch --show-current)"
echo ""

# Show what will be pushed
echo "Commits to be pushed:"
git log origin/develop..HEAD --oneline
echo ""

# Push the branch
echo "To push this branch, run one of the following commands:"
echo ""
echo "1. If you have GitHub CLI installed:"
echo "   gh auth login  # (if not already logged in)"
echo "   git push origin disco-paddle"
echo ""
echo "2. If you have SSH configured:"
echo "   git remote set-url origin git@github.com:micelvrice/PaddleNLP.git"
echo "   git push origin disco-paddle"
echo ""
echo "3. Using HTTPS with token:"
echo "   git push https://<your-github-username>:<your-token>@github.com/micelvrice/PaddleNLP.git disco-paddle"
echo ""
echo "4. Or simply:"
echo "   git push origin disco-paddle"
echo "   (You'll be prompted for credentials)"
echo ""
echo "After pushing, you can create a PR at:"
echo "https://github.com/PaddlePaddle/PaddleNLP/compare/develop...micelvrice:PaddleNLP:disco-paddle"