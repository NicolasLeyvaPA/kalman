import { Component } from 'react';
import { ErrorState } from './ErrorState';

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error) {
    // eslint-disable-next-line no-console
    console.error('ui error boundary caught:', error);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="max-w-2xl mx-auto mt-8">
          <ErrorState
            error={this.state.error}
            onRetry={() => this.setState({ error: null })}
          />
        </div>
      );
    }
    return this.props.children;
  }
}
