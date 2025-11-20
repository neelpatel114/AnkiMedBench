# Contributing to AnkiMedBench

Thank you for your interest in contributing to AnkiMedBench! This document provides guidelines for contributions.

## How to Contribute

### Reporting Issues

- Check existing issues before creating a new one
- Provide detailed description and reproduction steps
- Include system information (OS, Python version, GPU)
- Share error messages and logs

### Suggesting Enhancements

- Open an issue with the `enhancement` label
- Describe the proposed feature clearly
- Explain the use case and benefits
- Consider backward compatibility

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
   - Follow existing code style
   - Add tests if applicable
   - Update documentation
4. **Test your changes**
   ```bash
   python -m pytest tests/
   ```
5. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: description of feature"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Comment complex logic

## Adding New Benchmarks

To add a new benchmark task:

1. Create directory: `benchmarks/YourBenchmark/`
2. Implement benchmark script with standard interface
3. Add documentation in `docs/`
4. Update main README
5. Include example usage

## Adding Model Support

To add support for a new model family:

1. Create benchmark scripts in appropriate directories
2. Test with at least 2 model sizes
3. Document memory requirements
4. Add to README's supported models list

## Documentation

- Update README.md for user-facing changes
- Update docs/SETUP_GUIDE.md for setup changes
- Add inline code comments for complex logic
- Include examples in docstrings

## Testing

- Test on multiple model sizes if applicable
- Verify backward compatibility
- Check memory usage and performance
- Test on different datasets

## Pull Request Process

1. Update documentation as needed
2. Add your changes to a "Next Release" section in CHANGELOG
3. The PR will be reviewed by maintainers
4. Address any feedback
5. Once approved, it will be merged

## Code of Conduct

- Be respectful and professional
- Welcome newcomers
- Focus on constructive feedback
- Respect differing viewpoints

## Questions?

Open an issue with the `question` label or contact the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
