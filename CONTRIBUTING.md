# Contributing Guidelines

Thank you for your interest in contributing to the Medical Multimodal Chain-of-Thought framework!

## Important: License Notice

**Before contributing, please note:**

This project is under a custom restrictive license. All contributions become subject to the same license terms, particularly the restrictions on academic and commercial use without permission. By contributing, you agree to these terms.

See [LICENSE](LICENSE) for details.

## How to Contribute

### Reporting Issues

1. Check existing issues to avoid duplicates
2. Use the issue template if available
3. Provide detailed information:
   - Environment (OS, Python version, GPU)
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and logs

### Suggesting Enhancements

1. Open an issue with the "enhancement" label
2. Clearly describe the proposed feature
3. Explain the use case and benefits
4. Consider implementation complexity

### Code Contributions

#### Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/aoiheaven/EuGenAI.git
cd EuGenAI

# Setup environment
bash setup.sh
source .venv/bin/activate

# Install development dependencies
uv pip install -e ".[dev]"
```

#### Coding Standards

**Style Guide:**
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use meaningful variable and function names

**Code Quality Tools:**
```bash
# Format code with black
black src/

# Sort imports
isort src/

# Type checking
mypy src/

# Linting
flake8 src/
```

**Comments and Documentation:**
- ALL code must be in English (no Chinese in code/comments)
- Write clear docstrings for all functions and classes
- Use Google-style docstrings
- Add inline comments for complex logic

**Example:**
```python
def process_medical_image(image: np.ndarray, target_size: int = 512) -> torch.Tensor:
    """
    Process medical image for model input.
    
    Args:
        image: Input image as numpy array [H, W, C]
        target_size: Target size for resizing
        
    Returns:
        Processed image tensor [C, H, W]
        
    Raises:
        ValueError: If image dimensions are invalid
    """
    # Validate input dimensions
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D image, got shape {image.shape}")
    
    # Resize and normalize
    ...
    
    return tensor
```

#### Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_model.py
```

**Writing Tests:**
- Add tests for new features
- Maintain >80% code coverage
- Test edge cases and error conditions
- Use descriptive test names

```python
def test_model_forward_pass_with_valid_input():
    """Test that model produces expected output shape with valid input."""
    model = MedicalMultimodalCoT(img_size=512)
    batch_size = 2
    
    # Create dummy input
    images = torch.randn(batch_size, 3, 512, 512)
    ...
    
    # Forward pass
    outputs = model(...)
    
    # Assertions
    assert 'diagnosis_logits' in outputs
    assert outputs['diagnosis_logits'].shape[0] == batch_size
```

#### Commit Messages

Use conventional commit format:

```
type(scope): brief description

Detailed explanation of changes (optional)

Fixes #issue_number (if applicable)
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(model): add support for 3D medical images

Implement 3D convolution layers and modify attention mechanism
to handle volumetric data.

Fixes #42
```

```
fix(dataset): handle missing bounding box in reasoning steps

Previously crashed when bbox was None. Now defaults to [0,0,0,0].
```

#### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

3. **Ensure code quality**
   ```bash
   black src/
   isort src/
   flake8 src/
   mypy src/
   pytest tests/
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "feat(scope): description"
   git push origin feature/your-feature-name
   ```

5. **Open Pull Request**
   - Fill out the PR template
   - Link related issues
   - Request review from maintainers

6. **Address review comments**
   - Make requested changes
   - Push updates to the same branch

7. **Merge**
   - Maintainer will merge after approval
   - Delete your feature branch after merge

#### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] All functions have docstrings
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] Documentation updated (README, etc.)
- [ ] No commented-out code
- [ ] No debug print statements
- [ ] English-only in code/comments

### Documentation Contributions

Improvements to documentation are highly valued!

**Areas to contribute:**
- Fix typos or clarify existing docs
- Add examples and tutorials
- Improve code comments
- Translate documentation (maintain English and Chinese versions)

**For README updates:**
- Update both `README.md` (English) and `README_zh.md` (Chinese)
- Keep formatting consistent
- Add links to new sections

## Areas for Contribution

### High Priority

- [ ] Support for 3D medical images (CT/MRI volumes)
- [ ] Pre-trained model weights
- [ ] More comprehensive evaluation metrics
- [ ] Data augmentation strategies
- [ ] Web-based demo interface

### Medium Priority

- [ ] Multi-GPU training support
- [ ] DICOM file handling
- [ ] More visualization options
- [ ] Integration with medical image viewers
- [ ] Quantitative interpretability metrics

### Low Priority (Nice to Have)

- [ ] Additional model architectures
- [ ] Cross-validation utilities
- [ ] Hyperparameter tuning tools
- [ ] Mobile deployment support

## Community Guidelines

### Be Respectful

- Treat all contributors with respect
- Constructive criticism only
- Focus on the idea, not the person

### Be Patient

- Maintainers review PRs as time permits
- Complex features may require multiple review rounds
- Don't bump issues or PRs unnecessarily

### Be Collaborative

- Discuss major changes before implementing
- Help review others' contributions
- Share knowledge and expertise

## Getting Help

- **Questions:** Open a GitHub issue with "question" label
- **Chat:** [If you have a chat channel]
- **Email:** Contact via GitHub Issues (for sensitive matters)

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes for significant contributions
- Project documentation

## License Reminder

By contributing, you agree that your contributions will be licensed under the same custom restrictive license as the project. See [LICENSE](LICENSE) for full terms.

---

Thank you for contributing to advancing medical AI with explainability! 🙏

