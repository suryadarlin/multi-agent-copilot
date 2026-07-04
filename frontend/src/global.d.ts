declare module '*.css' {}

// Some TS configs in editors can miss JSX runtime types until deps are installed.
// This keeps the frontend buildable once `npm install` runs.
declare namespace JSX {
  interface IntrinsicElements {
    [elemName: string]: any
  }
}


