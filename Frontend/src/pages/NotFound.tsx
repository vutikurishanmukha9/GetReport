import { Link } from "react-router-dom";

const NotFound = () => {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-6">
      <div className="text-center">
        <p className="mb-3 font-mono text-xs uppercase tracking-[0.18em] text-primary">GetReport</p>
        <h1 className="mb-3 text-5xl font-display font-bold text-foreground">Page not found</h1>
        <p className="mb-6 text-sm text-muted-foreground">That destination does not exist or has moved.</p>
        <Link to="/" className="inline-flex h-10 items-center rounded-xl bg-primary px-5 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2">
          Return home
        </Link>
      </div>
    </div>
  );
};

export default NotFound;
