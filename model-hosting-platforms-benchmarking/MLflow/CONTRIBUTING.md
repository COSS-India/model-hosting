# Contributing to MLflow ASR Service

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported
2. Create a detailed bug report including:
   - Description of the issue
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Error messages and logs

### Suggesting Features

1. Check if the feature has been suggested before
2. Create a feature request with:
   - Clear description
   - Use case and motivation
   - Proposed implementation (if any)
   - Examples or mockups

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Follow code style guidelines
   - Add tests if applicable
   - Update documentation

4. **Test your changes**:
   ```bash
   # Run tests
   pytest tests/
   
   # Test the service
   ./test_curl.sh ta ctc
   ```

5. **Commit your changes**:
   ```bash
   git commit -m "Add: description of your changes"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**:
   - Provide clear description
   - Reference related issues
   - Request review from maintainers

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- Docker (optional, for testing)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd MLflow
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv mlflow
   source mlflow/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up HuggingFace token**:
   ```bash
   export HF_TOKEN="your_token"
   ```

5. **Log the model**:
   ```bash
   python log_model.py
   ```

## Code Style

### Python

- Follow PEP 8 style guide
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Example

```python
def process_audio(audio_bytes: bytes, sample_rate: int = 16000) -> torch.Tensor:
    """
    Process audio bytes into a PyTorch tensor.
    
    Args:
        audio_bytes: Raw audio bytes
        sample_rate: Target sample rate (default: 16000)
    
    Returns:
        PyTorch tensor of audio waveform
    """
    # Implementation
    pass
```

### Documentation

- Use docstrings for all functions and classes
- Follow Google or NumPy docstring style
- Update README.md for user-facing changes
- Add comments for complex logic

### Testing

- Write tests for new features
- Ensure existing tests pass
- Aim for good test coverage

## Project Structure

```
MLflow/
├── mlflow_asr.py      # Main model wrapper
├── log_model.py        # Model logging script
├── test_curl.sh        # Test script
├── Dockerfile          # Docker configuration
├── requirements.txt    # Dependencies
├── README.md           # Main documentation
├── docs/               # Additional documentation
└── tests/              # Test files (if added)
```

## Commit Messages

Use clear, descriptive commit messages:

- **Format**: `Type: Description`
- **Types**: `Add`, `Fix`, `Update`, `Remove`, `Refactor`, `Docs`
- **Examples**:
  - `Add: Support for batch processing`
  - `Fix: Audio resampling issue`
  - `Update: Documentation for Docker deployment`
  - `Refactor: Improve error handling`

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass (if applicable)
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Commit messages are clear

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
How was this tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
```

## Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your PR will be merged
4. Thank you for contributing!

## Areas for Contribution

### High Priority

- Performance optimizations
- Error handling improvements
- Documentation enhancements
- Test coverage

### Medium Priority

- Batch processing support
- Streaming audio support
- Additional language support
- Monitoring and metrics

### Low Priority

- UI improvements
- Example applications
- Tutorials and guides

## Questions?

- Open an issue for questions
- Check existing documentation
- Reach out to maintainers

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

**Thank you for contributing!** 🎉






