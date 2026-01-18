# Contributing to Personal Biology Twin

Thank you for your interest in contributing to Personal Biology Twin! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)
- [License](#license)

## 🤝 Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other contributors
- Help create a positive community

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- Git
- Docker (optional, for containerized development)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/personal-biology-twin.git
   cd personal-biology-twin
   ```

3. Set up the upstream remote:
   ```bash
   git remote add upstream https://github.com/Manavarya09/PERSONAL-BIOLOGY-TWIN.git
   ```

## 🛠️ Development Setup

### Local Development

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development tools
   ```

3. **Run the demo:**
   ```bash
   python run_demo.py
   ```

### Docker Development

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t biology-twin .
docker run -p 8000:8000 -p 8501:8501 biology-twin
```

## 💡 How to Contribute

### Types of Contributions

- **🐛 Bug fixes** - Fix existing issues
- **✨ New features** - Add new functionality
- **📚 Documentation** - Improve docs, tutorials, or examples
- **🧪 Tests** - Add or improve test coverage
- **🔧 Tooling** - Improve development tools or CI/CD
- **🎨 UI/UX** - Improve user interface and experience

### Contribution Process

1. **Choose an issue** or **create a new one** describing what you want to work on
2. **Fork the repository** and create a feature branch
3. **Make your changes** following the coding standards
4. **Write tests** for new functionality
5. **Update documentation** if needed
6. **Run the test suite** to ensure everything works
7. **Submit a pull request** with a clear description

## 🔄 Development Workflow

### Branch Naming

- `feature/description-of-feature`
- `bugfix/issue-description`
- `docs/update-documentation`
- `test/add-new-tests`

### Commit Messages

Follow conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat: add counterfactual simulation endpoint

- Add new API endpoint for intervention analysis
- Include proper error handling and validation
- Update API documentation

Closes #123
```

### Pull Request Process

1. **Create a PR** from your feature branch to `main`
2. **Fill out the PR template** with:
   - Clear title and description
   - Link to related issues
   - Screenshots for UI changes
   - Test results

3. **Code Review** - Address reviewer feedback
4. **Merge** - Squash and merge approved PRs

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_core.py -v

# Run with coverage
pytest --cov=biology_twin tests/

# Run integration tests
pytest tests/integration/
```

### Writing Tests

- Use `pytest` framework
- Place tests in `tests/` directory
- Follow naming convention: `test_*.py`
- Use descriptive test names and docstrings
- Test both success and failure cases

Example:
```python
def test_foundation_model_encode():
    """Test that foundation model can encode physiological signals."""
    model = FoundationModel(embedding_dim=128, input_dim=3)
    signals = {'hr': np.random.randn(10, 3)}

    embedding = model.encode(signals)

    assert embedding.shape == (10, 128)
    assert not np.isnan(embedding).any()
```

## 📚 Documentation

### Code Documentation

- Use docstrings for all public functions/classes
- Follow Google/NumPy docstring format
- Include type hints where possible

### Documentation Updates

- Update README.md for new features
- Add API documentation for new endpoints
- Update installation instructions if dependencies change
- Add examples for new functionality

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Clear title** describing the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, etc.)
- **Error messages** and stack traces
- **Screenshots** if applicable

### Feature Requests

For new features, please provide:

- **Clear description** of the proposed feature
- **Use case** and why it's needed
- **Implementation ideas** if you have them
- **Mockups** or examples if applicable

## 🔒 Security Considerations

Since this is a health-related AI system:

- **Never commit sensitive data** (PHI, credentials, etc.)
- **Use secure coding practices** (input validation, etc.)
- **Report security issues** via SECURITY.md guidelines
- **Follow HIPAA/privacy best practices** in health data handling

## 📋 Checklist for Contributions

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes
- [ ] No sensitive data committed
- [ ] License headers included for new files

## 🎯 Areas for Contribution

### High Priority
- Dataset integration improvements
- Model performance optimization
- Additional evaluation metrics
- Better error handling

### Research Areas
- Novel uncertainty quantification methods
- Advanced causal inference techniques
- Multi-modal physiological modeling
- Federated learning improvements

### Infrastructure
- CI/CD pipeline enhancements
- Docker optimization
- Monitoring and logging improvements
- API performance optimization

## 📞 Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Documentation**: Check the README and docs/ directory first

## 🙏 Recognition

Contributors will be recognized in:
- GitHub repository contributors list
- CHANGELOG.md for significant contributions
- Academic publications (where applicable)

Thank you for contributing to Personal Biology Twin! 🚀