# Integration Checklist

## ✅ Person 2 (Data Collection & Infrastructure) - COMPLETE

- [x] Metrics module
- [x] Vectorized environments  
- [x] Single environment collector
- [x] Vectorized collector
- [x] Evaluator
- [x] Value trajectory extraction
- [x] Logger
- [x] All tests passing (7/7)
- [x] Documentation written

**Status**: Ready for integration

---

## ⏳ Person 1 (Core Algorithm & Training)

### Before Integration
- [ ] Review `docs/PERSON2_INTERFACE.md`
- [ ] Understand collector interface
- [ ] Understand logger interface
- [ ] Run `scripts/quick_integration_test.py` (if available)

### Agent 0 (K=1, n=1)
- [ ] Import SingleEnvCollector
- [ ] Import Logger
- [ ] Import evaluate_agent
- [ ] Implement training loop
- [ ] Test that logs are created
- [ ] Verify CSV format is correct
- [ ] Train for 500k steps with 3 seeds
- [ ] Confirm agent reaches optimal policy (return=500)

### Agent 1 (K=1, n=1, stochastic)
- [ ] Add reward masking (already in wrappers?)
- [ ] Train for 500k steps with 3 seeds
- [ ] Verify value function converges correctly

### Agent 2 (K=6, n=1)
- [ ] Switch to RolloutCollector
- [ ] Handle final_states correctly (already a list)
- [ ] Train for 500k steps with 3 seeds

### Agent 3 (K=1, n=6)
- [ ] Configure n_steps=6
- [ ] Train for 500k steps with 3 seeds

### Agent 4 (K=6, n=6)
- [ ] Use RolloutCollector with n_steps=6
- [ ] Train for 500k steps with 3 seeds

### Verification
- [ ] All agents complete training
- [ ] All logs are created correctly
- [ ] No crashes or errors
- [ ] Checkpoints saved (optional)

---

## ⏳ Person 3 (Plotting & Analysis)

### Setup
- [ ] Review `docs/PERSON2_INTERFACE.md`
- [ ] Understand CSV data format
- [ ] Understand directory structure
- [ ] Test data loading functions

### Data Loading
- [ ] Can load `train_log.csv`
- [ ] Can load `eval_log.csv`
- [ ] Can load value trajectories
- [ ] Can aggregate across seeds

### Required Plots (from PDF)

#### Individual Agent Plots
For each agent (0, 1, 2, 3, 4):
- [ ] Training return curve (with min/max bands)
- [ ] Evaluation return curve (with min/max bands)
- [ ] Actor loss curve
- [ ] Critic loss curve
- [ ] Value trajectory evolution (mean over trajectory)
- [ ] Value trajectory (individual values)

#### Comparison Plots
- [ ] All agents training returns on one plot
- [ ] All agents evaluation returns on one plot
- [ ] Learning speed comparison (time to reach 450+ return)
- [ ] Stability comparison (variance in returns)

### Analysis Questions (from PDF)

#### Agent 0
- [ ] Plot value function after convergence
- [ ] Answer: What values does V(s) take with correct bootstrapping?
- [ ] Answer: What happens without correct bootstrapping?

#### Agent 1
- [ ] Answer: What value does V(s) take after convergence?
- [ ] Answer: How does value loss compare to deterministic env?
- [ ] Answer: How stable is learning compared to deterministic?

#### Agent 2
- [ ] Answer: Is learning faster/slower than K=1 (steps vs wall-time)?
- [ ] Answer: Is learning more/less stable than K=1?

#### Agent 3
- [ ] Answer: What value does V(s) converge to?
- [ ] Answer: Is learning faster/slower than n=1?
- [ ] Answer: Is learning more/less stable than n=1?
- [ ] Answer: What does algorithm resemble if n>500?

#### Agent 4
- [ ] Answer: Is learning faster/slower than K=1,n=1?
- [ ] Answer: Is learning more/less stable?
- [ ] Answer: Can you increase learning rates while staying stable?

### Final Deliverables
- [ ] All plots generated
- [ ] All questions answered
- [ ] Notebook complete with analysis
- [ ] Video presentation recorded

---

## 🤝 Communication Log

### Person 2 → Team
- [ ] "All modules complete and tested"
- [ ] "Check docs/PERSON2_INTERFACE.md"
- [ ] "Ready for integration"

### Person 1 → Person 2
- [ ] Questions about collector usage (if any)
- [ ] Questions about logging (if any)
- [ ] Confirmation that integration works

### Person 3 → Person 2
- [ ] Questions about data format (if any)
- [ ] Request for sample data (if needed)
- [ ] Confirmation that data loads correctly

### Team → Team
- [ ] Progress updates (daily standups?)
- [ ] Issue tracking
- [ ] Final submission coordination

---

## 🚨 Common Issues

### Issue: "Module not found"
**Solution**: Run from project root, or add to PYTHONPATH

### Issue: "Truncation not working"
**Solution**: Check that `truncated` flag is separate from `done` flag

### Issue: "Episodes too short"
**Solution**: Agent is untrained. Keep training.

### Issue: "Value function looks wrong"
**Solution**: Verify bootstrapping logic in returns.py

### Issue: "Data doesn't align across seeds"
**Solution**: Use `step` column, not row index

---

## ⏰ Timeline

- **Day 1**: Person 1 completes Agent 0
- **Day 2**: Person 1 completes Agents 1-4
- **Day 3**: Person 3 creates all plots
- **Day 4**: Analysis, notebook, video
- **Day 5**: Buffer & submission

**Deadline**: Tuesday Jan 13, 9:00 AM

---

**Last Updated**: January 11, 2026