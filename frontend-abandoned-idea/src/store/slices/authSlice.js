import { createSlice } from '@reduxjs/toolkit';
import { auth } from '../../firebase';
import { 
  signInWithEmailAndPassword, 
  createUserWithEmailAndPassword, 
  signInWithPopup,
  GoogleAuthProvider,
  signOut 
} from 'firebase/auth';

const initialState = {
  user: null,
  isAuthenticated: false,
  loading: false,
  error: null,
};

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    loginStart: (state) => {
      state.loading = true;
      state.error = null;
    },
    loginSuccess: (state, action) => {
      state.loading = false;
      state.isAuthenticated = true;
      state.user = {
        ...action.payload,
        balance: action.payload.balance || 1000, // Default balance for new users
      };
    },
    loginFailure: (state, action) => {
      state.loading = false;
      state.error = action.payload;
    },
    logout: (state) => {
      state.user = null;
      state.isAuthenticated = false;
    },
    updateBalance: (state, action) => {
      if (state.user) {
        state.user.balance = action.payload;
      }
    },
  },
});

export const { loginStart, loginSuccess, loginFailure, logout, updateBalance } = authSlice.actions;

// Thunk actions for Firebase authentication
export const signInWithEmail = (email, password) => async (dispatch) => {
  try {
    dispatch(loginStart());
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    dispatch(loginSuccess({
      id: userCredential.user.uid,
      email: userCredential.user.email,
      name: userCredential.user.displayName || email.split('@')[0],
    }));
  } catch (error) {
    dispatch(loginFailure(error.message));
  }
};

export const signUpWithEmail = (email, password) => async (dispatch) => {
  try {
    dispatch(loginStart());
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    dispatch(loginSuccess({
      id: userCredential.user.uid,
      email: userCredential.user.email,
      name: email.split('@')[0],
    }));
  } catch (error) {
    dispatch(loginFailure(error.message));
  }
};

export const signInWithGoogle = () => async (dispatch) => {
  try {
    dispatch(loginStart());
    const provider = new GoogleAuthProvider();
    const userCredential = await signInWithPopup(auth, provider);
    dispatch(loginSuccess({
      id: userCredential.user.uid,
      email: userCredential.user.email,
      name: userCredential.user.displayName,
    }));
  } catch (error) {
    dispatch(loginFailure(error.message));
  }
};

export const logoutUser = () => async (dispatch) => {
  try {
    await signOut(auth);
    dispatch(logout());
  } catch (error) {
    console.error('Logout error:', error);
  }
};

export default authSlice.reducer;